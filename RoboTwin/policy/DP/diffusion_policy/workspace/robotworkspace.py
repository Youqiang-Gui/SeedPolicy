if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import datetime
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, Sampler
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ====================================================================
# 1. 移植核心组件：WrappedDataset, Sampler, Collate
# ====================================================================

class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        idx, is_reset = item
        data = self.dataset[idx]
        data['is_reset'] = is_reset
        return data


class StaggeredEpisodeBatchSampler(Sampler):
    def __init__(
            self,
            dataset,
            batch_size: int,
            step_size: int = 6,
            seed: int = 0,
            max_reset_interval: int = 8,  # L
            min_reset_interval: int = 1,  # ⭐新增：最小 l
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_reset_interval = max_reset_interval
        self.min_reset_interval = min_reset_interval
        self.rng = np.random.default_rng(seed)

        # 1. 获取所有 Episode 的边界
        episode_ids = dataset.get_sampler_episode_ids()
        self.episode_starts = []

        diff = np.diff(episode_ids)
        change_indices = np.where(diff != 0)[0] + 1

        start = 0
        for end in change_indices:
            self.episode_starts.append((start, end))
            start = end
        self.episode_starts.append((start, len(episode_ids)))

        # 2. 生成所有 offset 链任务
        self.all_chain_tasks = []
        for ep_start, ep_end in self.episode_starts:
            ep_len = ep_end - ep_start
            for offset in range(self.step_size):
                if ep_start + offset < ep_end:
                    # 计算 offset 链长度
                    max_chain_len = ((ep_len - offset - 1) // self.step_size) + 1

                    self.all_chain_tasks.append({
                        'start': ep_start,
                        'end': ep_end,
                        'offset': offset,
                        'max_chain_len': max_chain_len,
                    })

        self.num_tasks = len(self.all_chain_tasks)
        
        total_samples = len(dataset)
        self.num_batches = total_samples // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):

        tasks = self.all_chain_tasks.copy()
        self.rng.shuffle(tasks)
        task_queue = iter(tasks)

        # 随机产生合法 reset 区间 [min, max]
        def sample_reset_interval(max_chain_len):
            hi = min(self.max_reset_interval, max_chain_len)
            lo = self.min_reset_interval

            if lo > hi:
                return lo

            return self.rng.integers(lo, hi + 1)

        # 填充一个 slot
        def fill_slot(slot_idx):
            nonlocal tasks, task_queue

            try:
                task = next(task_queue)
            except StopIteration:
                self.rng.shuffle(tasks)
                task_queue = iter(tasks)
                task = next(task_queue)

            first_idx = task['start'] + task['offset']
            max_chain_len = task['max_chain_len']

            # 随机 reset 间隔
            reset_interval = sample_reset_interval(max_chain_len)

            return {
                'current_idx': first_idx,
                'end': task['end'],
                'step': self.step_size,

                # reset 控制
                'just_reset': True,
                'steps_since_reset': 0,
                'reset_interval': reset_interval,
                'max_chain_len': max_chain_len,
            }

        # 初始化所有 slot
        slots = [fill_slot(i) for i in range(self.batch_size)]

        # 生成 batch
        for _ in range(self.num_batches):
            batch_indices = []

            for i in range(self.batch_size):
                slot = slots[i]
                idx = slot['current_idx']

                is_reset = slot['just_reset']
                batch_indices.append((idx, is_reset))

                slot['just_reset'] = False
                slot['current_idx'] += slot['step']
                slot['steps_since_reset'] += 1

                # episode 结束
                if slot['current_idx'] >= slot['end']:
                    slots[i] = fill_slot(i)
                    continue

                # 达到 reset 区间
                if slot['steps_since_reset'] >= slot['reset_interval']:
                    slot['just_reset'] = True
                    slot['steps_since_reset'] = 0
                    slot['reset_interval'] = sample_reset_interval(slot['max_chain_len'])

            yield batch_indices

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def stateful_collate_safe(batch_of_samples):
    is_reset_list = [s['is_reset'] for s in batch_of_samples]
    is_reset = torch.tensor(is_reset_list, dtype=torch.bool)

    clean_samples = []
    for s in batch_of_samples:
        s_copy = s.copy()
        del s_copy['is_reset']
        clean_samples.append(s_copy)

    collated_batch = {}
    for key in clean_samples[0].keys():
        collated_batch[key] = torch.stack([s[key] for s in clean_samples])

    return collated_batch, is_reset

# ====================================================================
# 2. 移植 Dataloader 创建函数
# ====================================================================

def create_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
    # 下面这些参数如果 config 里没有，会使用默认值
    step_size: int = 6, 
    min_reset_interval: int = 1,
    max_reset_interval: int = 10,
    # 兼容性参数，这里用不到但防止报错
    shuffle: bool = False, 
    pin_memory: bool = True,
    persistent_workers: bool = False,
):
    print("\n" + "=" * 50)
    print("Using Staggered Chain Sampler (Stateful)")
    print(f"Batch Size: {batch_size}, Step Size: {step_size}")
    print(f"Reset interval: [{min_reset_interval}, {max_reset_interval}]")
    print("=" * 50 + "\n")

    batch_sampler = StaggeredEpisodeBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        step_size=step_size,
        seed=seed,
        max_reset_interval=max_reset_interval,
        min_reset_interval=min_reset_interval,
    )

    wrapped_dataset = WrappedDataset(dataset)

    dataloader = DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=stateful_collate_safe,
        worker_init_fn=seed_worker,
    )
    return dataloader


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # 注意：确保你的 cfg.policy 指向的模型支持 (batch, hidden_state) 输入
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        def count_parameters(model):
            # 计算总参数量
            total_params = sum(p.numel() for p in model.parameters())
            # 计算可训练参数量
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params

        total, trainable = count_parameters(self.model)
        print("-" * 30)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total Parameters: {total:,} ({total / 1e6:.2f} M)")
        print(f"Trainable Parameters: {trainable:,} ({trainable / 1e6:.2f} M)")
        print("-" * 30)
        # -----------------------

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
        self.ckpt_relpath = f"checkpoints/{save_name}-{seed}-{cfg.config_name}/{self.timestamp}"
        os.makedirs(self.ckpt_relpath, exist_ok=True)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # ==========================================================
        # 限制 worker 数量，防止全内存数据集导致的 OOM
        # ==========================================================
        safe_num_workers = min(cfg.dataloader.num_workers, 8)
        print(f"Config num_workers: {cfg.dataloader.num_workers}, Clamped to: {safe_num_workers}")
        
        # 传递额外的 Sampler 参数 (如果 config 里有就用，没有就用默认)
        dataloader_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        dataloader_kwargs['num_workers'] = safe_num_workers
        # 移除可能导致冲突的参数（如果 create_dataloader 不支持）
        if 'shuffle' in dataloader_kwargs: del dataloader_kwargs['shuffle']
        
        # 手动注入一些默认参数，或者从 cfg.training 里取（如果你的 config 结构不同请调整）
        # 这里假设 step_size 等参数可能在 cfg.task.dataset 或者直接硬编码，或者作为 dataloader 的一部分
        # 为了演示，这里直接使用默认值调用，你可以通过 cfg 传递
        train_dataloader = create_dataloader(
            dataset, 
            seed=seed,
            step_size=6, # 可以改为 cfg.training.get('step_size', 6)
            min_reset_interval=40,
            max_reset_interval=40,
            **dataloader_kwargs
        )
        
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        val_dataloader_kwargs['num_workers'] = 0 # 验证集建议 0 workers
        if 'shuffle' in val_dataloader_kwargs: del val_dataloader_kwargs['shuffle']
        
        val_dataloader = create_dataloader(
            val_dataset,
            seed=seed,
            step_size=6,
            min_reset_interval=40,
            max_reset_interval=40,
            **val_dataloader_kwargs
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # 初始化 Hidden State
                hidden_state = None

                if getattr(cfg.training, "freeze_encoder", False) and hasattr(self.model, "obs_encoder"):
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                # ========= train for this epoch ==========
                train_losses = list()
                self.model.train() # 确保在 train 模式

                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    # 注意：DataLoader 返回的是 (batch_dict, is_reset)
                    for batch_idx, (batch_from_collate, is_reset) in enumerate(tepoch):
                        
                        # Postprocess
                        batch = dataset.postprocess(batch_from_collate, device)
                        
                        # 处理 is_reset 并移动到 GPU
                        if isinstance(is_reset, list):
                            is_reset = torch.tensor(is_reset, dtype=torch.bool)
                        batch['is_reset'] = is_reset.to(device)

                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss (Stateful)
                        # 模型需要返回 (loss, new_state)
                        raw_loss, state_with_pos = self.model.compute_loss(batch, hidden_state)
                        
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        # Detach hidden state for next step
                        hidden_state = state_with_pos.detach()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_hidden_state = None # Validation 也要维护 State
                        
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, (batch_from_collate, is_reset) in enumerate(tepoch):
                                batch = dataset.postprocess(batch_from_collate, device)
                                
                                if isinstance(is_reset, list):
                                    is_reset = torch.tensor(is_reset, dtype=torch.bool)
                                batch['is_reset'] = is_reset.to(device)

                                # Stateful Validation
                                loss, new_val_hidden_state = self.model.compute_loss(batch, val_hidden_state)
                                val_hidden_state = new_val_hidden_state
                                
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        # Sampling 也需要考虑 State，这里传入 None 表示从头开始
                        # 如果模型支持 state 参数
                        result = policy.predict_action(obs_dict, state=None)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    ckpt_relpath = f"checkpoints/{save_name}-{seed}-{cfg.config_name}/{self.timestamp}/{self.epoch + 1}.ckpt"
                    self.save_checkpoint(ckpt_relpath)

                # ========= eval end for this epoch ==========
                policy.train()
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()