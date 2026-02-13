from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action"],
        )
        print("="*50)
        print("Pre-loading entire dataset into RAM...")
        self.in_memory_data = dict()
        for key in self.replay_buffer.keys():
            # 使用 [:] 切片操作，这是 zarr/numpy 加载整个数组到内存的标准方法
            print(f"Loading {key}...")
            self.in_memory_data[key] = self.replay_buffer[key][:]
        print("Dataset pre-loading complete!")
        print("="*50)

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_sampler_episode_ids(self) -> np.ndarray: # <--- 新增方法
        """返回采样器中每个有效序列对应的 episode ID"""
        return self.sampler.episode_ids

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.in_memory_data["action"],
            "agent_pos": self.in_memory_data["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, (int, np.integer)):
            # 获取“取数指令”
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
            
            sample_result = dict()
            # 遍历数据键
            for key in self.in_memory_data.keys():
                # 从内存中的 NumPy 数组读取数据！
                input_arr = self.in_memory_data[key]
                sample = input_arr[buffer_start_idx:buffer_end_idx]
                
                data = sample
                # 处理填充（padding），使用 self.horizon
                if (sample_start_idx > 0) or (sample_end_idx < self.horizon): # <--- 修改：使用 self.horizon
                    data = np.zeros(
                        shape=(self.horizon,) + input_arr.shape[1:], # <--- 修改：使用 self.horizon
                        dtype=input_arr.dtype,
                    )
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.horizon: # <--- 修改：使用 self.horizon
                        data[sample_end_idx:] = sample[-1]
                    data[sample_start_idx:sample_end_idx] = sample
                sample_result[key] = data

            sample = dict_apply(sample_result, torch.from_numpy)
            return sample

        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            # 循环 self.in_memory_data 而不是 replay_buffer
            for k, v in self.in_memory_data.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v, # v 现在是内存中的完整 NumPy 数组
                    self.sampler.indices,
                    idx,
                    self.horizon, # <--- 修改：使用 self.horizon
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                # 'front_cam': front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
