from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

# =============================================================================
# 1. 状态更新相关的辅助模块 (从参考代码迁移)
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class CrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (query_dim // num_heads) ** -0.5
        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(kv_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, return_attn=True):
        B, Nq, Cq = query.shape
        _, Nk, Ck = key.shape
        head_dim = Cq // self.num_heads

        q = self.q(query).reshape(B, Nq, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, Nk, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, Nk, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # 1. 计算原始分数 (Raw Scores)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # 2. 如果需要返回 attention，我们保存原始分数 attn_scores，而不是 softmax 后的结果
        # 参考代码逻辑：我们要衡量的是"匹配强度"，而不是"概率分布的均值"
        raw_attn_for_gate = attn_scores if return_attn else None

        # 3. 计算 Softmax 用于加权 Value
        attn_probs = attn_scores.softmax(dim=-1) 
        attn_probs = self.attn_drop(attn_probs)
        
        x = (attn_probs @ v).transpose(1, 2).reshape(B, Nq, Ck)
        x = self.proj_drop(self.proj(x))
        
        if return_attn:
            return x, raw_attn_for_gate # <--- 修改这里：返回 raw_attn_for_gate
        return x

class StateUpdaterBlock(nn.Module):
    def __init__(self, state_dim: int, obs_feat_dim: int, n_head: int = 8, mlp_ratio: float = 4.0, dropout: float = 0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(state_dim)
        self.attn = SelfAttention(state_dim, num_heads=n_head, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        
        self.norm2 = nn.LayerNorm(state_dim)
        self.norm_mem = nn.LayerNorm(obs_feat_dim)
        self.cross_attn = CrossAttention(query_dim=state_dim, kv_dim=obs_feat_dim, num_heads=n_head, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)

        self.norm3 = nn.LayerNorm(state_dim)
        self.mlp = Mlp(in_features=state_dim, hidden_features=int(state_dim * mlp_ratio), drop=dropout)
        self.drop_path = nn.Identity()

    def forward(self, state: torch.Tensor, obs_feat: torch.Tensor) -> torch.Tensor:
        state = state + self.drop_path(self.attn(self.norm1(state)))
        
        obs_feat_normed = self.norm_mem(obs_feat)
        cross_attn_output, cross_attn_weights = self.cross_attn(self.norm2(state), obs_feat_normed, obs_feat_normed, return_attn=True)
        state = state + self.drop_path(cross_attn_output)
        
        state = state + self.drop_path(self.mlp(self.norm3(state)))
        return state, cross_attn_weights

class ParallelStateUpdater(nn.Module):
    def __init__(self, depth: int, state_dim: int, obs_feat_dim: int, **block_kwargs):
        super().__init__()
        # 更新 State 的流
        self.state_blocks = nn.ModuleList([
            StateUpdaterBlock(state_dim=state_dim, obs_feat_dim=obs_feat_dim, **block_kwargs) 
            for _ in range(depth)
        ])
        # 更新 Obs Feature 的流 (注意维度互换)
        self.obs_blocks = nn.ModuleList([
            StateUpdaterBlock(state_dim=obs_feat_dim, obs_feat_dim=state_dim, **block_kwargs) 
            for _ in range(depth)
        ])
        self.final_norm_state = nn.LayerNorm(state_dim)
        self.final_norm_obs = nn.LayerNorm(obs_feat_dim)

    def forward(self, state: torch.Tensor, obs_feat: torch.Tensor):
        all_cross_attn_weights = []
        for state_block, obs_block in zip(self.state_blocks, self.obs_blocks):
            new_state, state_cross_attn = state_block(state, obs_feat)
            new_obs, obs_cross_attn = obs_block(obs_feat, state)
            
            state = new_state
            obs_feat = new_obs
            all_cross_attn_weights.append(state_cross_attn)
        
        return self.final_norm_state(state), self.final_norm_obs(obs_feat), all_cross_attn_weights


# =============================================================================
# 2. 修改后的 DiffusionUnetImagePolicy
# =============================================================================

class DiffusionUnetImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # State related params
        state_dim=512,
        num_state_patches=60,
        state_updater_depth=6,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        # get feature dim from the existing encoder
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        # 注意：这里我们强制要求使用 obs_as_global_cond，因为 state updater 的输出更适合做 global cond
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        
        # 强制修正：我们将 Updater 输出的特征展平作为 Global Cond
        # Updater 输出 shape: (B, n_obs_steps, obs_feature_dim)
        # Flatten 后: obs_feature_dim * n_obs_steps
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = state_dim * n_obs_steps
        else:
             # 如果你坚持用 local cond，逻辑会很复杂，这里暂时建议配合 global cond 使用
             raise NotImplementedError("Stateful training currently assumes obs_as_global_cond=True")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # --- State Logic Initialization ---
        self.state_dim = state_dim
        self.num_state_patches = num_state_patches
        
        # 可学习的初始状态
        self.initial_state_tokens = nn.Parameter(
            torch.randn(1, self.num_state_patches, self.state_dim) * 0.02
        )
        # 状态的位置编码
        self.state_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_state_patches, self.state_dim)
        )
        
        # 双流状态更新器
        self.state_updater = ParallelStateUpdater(
            depth=state_updater_depth,
            state_dim=self.state_dim,
            obs_feat_dim=self.state_dim,
            n_head=8,
            mlp_ratio=4,
            dropout=0
        )
        print(f"Initialized Stateful Policy with State Dim: {state_dim}, Obs Dim: {obs_feature_dim}")

        # self.obs_feature_proj = nn.Linear(obs_feature_dim, self.state_dim)
        # self.obs_feature_norm = nn.LayerNorm(self.state_dim)

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        state: previous hidden state (B, N, D). If None, will use initial state.
        result: includes "action", "action_pred", "state"
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        # Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # 1. Encode Observations
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (B*To, Do)
        # nobs_features = self.obs_feature_proj(nobs_features)
        # nobs_features = self.obs_feature_norm(nobs_features)
        nobs_features = nobs_features.reshape(B, To, -1) # (B, To, Do)

        # 2. State Update Logic (Inference Mode)
        is_first_step = (state is None)
        if is_first_step:
            state_content = self.initial_state_tokens.expand(B, -1, -1)
        else:
            state_content = state
        
        state_with_pos = state_content + self.state_pos_embed
        
        # Run Updater
        # new_state_with_pos: (B, N_state, D_state)
        # updated_obs: (B, To, Do)
        new_state_with_pos, updated_obs, all_cross_attn = self.state_updater(
            state_with_pos, 
            nobs_features
        )
        
        # Soft Gating
        if is_first_step:
            update_gate = torch.ones(B, 1, 1, device=self.device)
        else:
            stacked_attn = torch.stack(all_cross_attn, dim=0)
            rearranged_attn = rearrange(stacked_attn, 'l b h n_state n_obs -> b n_state n_obs (l h)')
            state_query_img_key = rearranged_attn.mean(dim=(-1, -2))
            update_gate = torch.sigmoid(state_query_img_key).unsqueeze(-1)

        final_updated_state_with_pos = new_state_with_pos * update_gate + state_with_pos * (1 - update_gate)
        
        # Prepare next state (remove pos embed)
        updated_state_content = final_updated_state_with_pos - self.state_pos_embed

        # 3. Prepare UNet Conditions
        # Use updated_obs as global condition
        global_cond = updated_obs.reshape(B, -1) # Flatten (B, To*Do)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action, 
            "action_pred": action_pred,
            "state": updated_state_content # Return new state
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, state=None):
        """
        batch: input batch
        state: previous hidden state (B, N, D). Can be None.
        """
        # normalize input
        assert "valid_mask" not in batch
        is_reset = batch.get('is_reset') # 获取 reset 标志
        
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        # horizon = nactions.shape[1]

        # 1. Encode Observations
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (B*To, Do)
        # nobs_features = self.obs_feature_proj(nobs_features)
        # nobs_features = self.obs_feature_norm(nobs_features)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, -1) # (B, To, Do)

        # 2. State Update Logic
        initial_state = self.initial_state_tokens.expand(batch_size, -1, -1)
        old_state = initial_state if state is None else state
        
        # 根据 is_reset 决定是否重置 State
        # is_reset shape: (B,) -> (B, 1, 1)
        state_in = torch.where(is_reset.view(batch_size, 1, 1), initial_state, old_state)
        state_with_pos = state_in + self.state_pos_embed

        # Run Updater
        updated_state_with_pos, updated_obs, all_cross_attn = self.state_updater(
            state_with_pos, 
            nobs_features
        )
        
        # Calculate Gate
        stacked_attn = torch.stack(all_cross_attn, dim=0)
        rearranged_attn = rearrange(stacked_attn, 'l b h n_state n_obs -> b n_state n_obs (l h)')
        state_query_img_key = rearranged_attn.mean(dim=(-1, -2))
        calculated_gate = torch.sigmoid(state_query_img_key).unsqueeze(-1)
        
        # 如果是 reset 步，强制 gate 为 1 (完全采纳新状态/其实是重置)
        update_gate = torch.where(is_reset.view(batch_size, 1, 1), torch.tensor(1.0, device=self.device), calculated_gate)

        # Apply Gate
        final_state_with_pos = updated_state_with_pos * update_gate + state_with_pos * (1 - update_gate)

        # 3. Prepare Conditions for UNet
        # 使用更新后的 Observation Feature 作为 Global Condition
        global_cond = updated_obs.reshape(batch_size, -1) # Flatten

        trajectory = nactions
        cond_data = trajectory

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        # 传入 global_cond (来自 state updater)
        pred = self.model(noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        
        # 返回 loss 和 更新后的 state (去掉 pos embed, 带梯度)
        # 注意：Training Loop 那边需要负责 .detach()
        new_state = final_state_with_pos - self.state_pos_embed
        
        return loss, new_state