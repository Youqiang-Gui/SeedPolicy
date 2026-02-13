from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules



# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         mlp_hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, embed_dim)
#         )

#     def forward(self, x):
#         # Self-attention part
#         x_norm1 = self.norm1(x)
#         attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
#         x = x + attn_output # Residual connection
        
#         # MLP part
#         x_norm2 = self.norm2(x)
#         mlp_output = self.mlp(x_norm2)
#         x = x + mlp_output # Residual connection
#         return x


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding """
#     def __init__(self, img_size=(240, 320), patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

# class VisionTransformerEncoder(nn.Module):
#     """
#     新的观测编码器，使用 Patch Embedding 并融合多帧信息。
#     【已修复维度错误】
#     """
#     def __init__(self,
#                  n_obs_steps: int,
#                  img_size: tuple = (240, 320),
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  agent_pos_dim: int = 14,
#                  num_transformer_layers = 1,
#                  num_heads = 8,
#                  out_dim: int = 256):
#         super().__init__()
#         self.n_obs_steps = n_obs_steps
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         self.num_patches = self.patch_embed.num_patches
#         self.image_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

#         # ==================== 【修复核心】 ====================
#         # 使用一个简单的线性层在时间维度上进行融合。
#         # 它将 n_obs_steps (例如3) 个时间点的特征投影为1个。
#         self.time_fusion_layer = nn.Linear(n_obs_steps, 1)

#         self.transformer_blocks = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads)
#             for _ in range(num_transformer_layers)
#         ])
#         # ====================================================

#         # 最终的投影层
#         self.projection = nn.Linear(embed_dim + agent_pos_dim, out_dim)
#         self.agent_pos_dim = agent_pos_dim
#         self.out_dim = out_dim

#     def forward(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         # nobs['head_cam']: (B * To, C, H, W)
#         # nobs['agent_pos']: (B * To, Da)
        
#         # 1. 图像 Patch Embedding
#         img_embed = self.patch_embed(nobs['head_cam'])  # (B * To, num_patches, embed_dim=768)
#         img_embed = img_embed + self.image_pos_embed
#         for block in self.transformer_blocks:
#             img_embed = block(img_embed)
#         B_To, N, D_img = img_embed.shape
#         B = B_To // self.n_obs_steps
        
#         # 2. 时间融合
#         # a. 融合图像特征
#         img_embed_reshaped = img_embed.view(B, self.n_obs_steps, N, D_img) # (B, To, N, D_img)
#         # 将时间维度(To)移动到最后，以便线性层可以处理它
#         img_to_fuse = img_embed_reshaped.permute(0, 2, 3, 1) # (B, N, D_img, To)
#         # 使用线性层进行融合
#         fused_img_feat = self.time_fusion_layer(img_to_fuse).squeeze(-1) # (B, N, D_img)

#         # b. 融合 agent_pos
#         agent_pos = nobs['agent_pos'].view(B, self.n_obs_steps, -1) # (B, To, D_pos=14)
#         # 将时间维度(To)移动到最后
#         agent_pos_to_fuse = agent_pos.permute(0, 2, 1) # (B, D_pos, To)
#         fused_agent_pos = self.time_fusion_layer(agent_pos_to_fuse).squeeze(-1) # (B, D_pos)

#         # 3. 拼接 agent_pos 和图像特征
#         agent_pos_expanded = fused_agent_pos.unsqueeze(1).expand(-1, N, -1) # (B, N, D_pos)
#         combined_features = torch.cat([fused_img_feat, agent_pos_expanded], dim=-1) # (B, N, D_img + D_pos)
        
#         # 4. 最终投影
#         output = self.projection(combined_features) # (B, N, out_dim=256)
        
#         return output
    


class Mlp(nn.Module):
    """一个标准的前馈网络 (Feed-Forward Network)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    """标准的多头自注意力模块"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

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
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    """交叉注意力模块，适配 state (query) 和 obs_feat (kv) 的维度"""
    def __init__(self, query_dim, kv_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim_q = query_dim // num_heads
        self.scale = head_dim_q ** -0.5

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_dim, kv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 最终的投影层，将多头注意力的结果融合回 query_dim
        self.proj = nn.Linear(kv_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, return_attn=True):
        B, Nq, Cq = query.shape
        _, Nk, Ck = key.shape
        
        head_dim_kv = Ck // self.num_heads

        q = self.q(query).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, Nk, self.num_heads, head_dim_kv).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, Nk, self.num_heads, head_dim_kv).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn if return_attn else None

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Ck)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 【修改】返回注意力图
        if return_attn:
            return x, attn_weights
        else:
            return x
    


class StateUpdaterBlock(nn.Module):
    
    def __init__(self, state_dim: int, obs_feat_dim: int, n_head: int = 8, mlp_ratio: float = 4.0, dropout: float = 0, drop_path: float = 0.0):
        super().__init__()
        
        # === Self-Attention Path ===
        self.norm1 = nn.LayerNorm(state_dim)
        self.attn = SelfAttention(
            dim=state_dim,
            num_heads=n_head,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        # === Cross-Attention Path ===
        self.norm2 = nn.LayerNorm(state_dim)
        self.norm_mem = nn.LayerNorm(obs_feat_dim) # 对 memory (obs_feat) 进行 LayerNorm
        self.cross_attn = CrossAttention(
            query_dim=state_dim,
            kv_dim=obs_feat_dim,
            num_heads=n_head,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        # === MLP Path ===
        self.norm3 = nn.LayerNorm(state_dim)
        self.mlp = Mlp(
            in_features=state_dim,
            hidden_features=int(state_dim * mlp_ratio),
            drop=dropout
        )
        
        # DropPath 用于实现随机深度 (Stochastic Depth)，可以增强正则化
        # 在这里我们先用 Identity，如果需要可以增加 drop_path 概率
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path) # DropPath is not a standard module

    def forward(self, state: torch.Tensor, obs_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): 上一时刻的状态 (对应x), shape (B, N, Ds)
            obs_feat (torch.Tensor): 当前观测特征 (对应y), shape (B, N, Do)
            假设 state 和 obs_feat 都已经添加了各自的位置编码。

        Returns:
            torch.Tensor: 更新后的状态, shape (B, N, Ds)
        """
        # 1. Self-Attention on state
        state = state + self.drop_path(self.attn(self.norm1(state)))
        
        # 2. Cross-Attention
        obs_feat_normed = self.norm_mem(obs_feat)
        
        # 【修改】接收交叉注意力图
        cross_attn_output, cross_attn_weights = self.cross_attn(
            self.norm2(state), 
            obs_feat_normed, 
            obs_feat_normed, 
            return_attn=True # 总是请求注意力图
        )
        state = state + self.drop_path(cross_attn_output)
        
        # 3. MLP
        state = state + self.drop_path(self.mlp(self.norm3(state)))
        
        # 【修改】返回更新后的状态和注意力图
        return state, cross_attn_weights
    

class ParallelStateUpdater(nn.Module):
    """
    并行的双流状态更新器。
    它包含两组并行的 StateUpdaterBlock 列表，一组用于更新 state，
    另一组用于更新 obs_feat。
    """
    def __init__(self, depth: int, state_dim: int, obs_feat_dim: int, **block_kwargs):
        super().__init__()
        
        # 流1：用于更新 state (state 作为 Query, obs_feat 作为 Key/Value)
        self.state_blocks = nn.ModuleList([
            StateUpdaterBlock(state_dim=state_dim, obs_feat_dim=obs_feat_dim, **block_kwargs) 
            for _ in range(depth)
        ])
        
        # 流2：用于更新 obs_feat (obs_feat 作为 Query, state 作为 Key/Value)
        # 注意这里的维度参数是反过来的！
        self.obs_blocks = nn.ModuleList([
            StateUpdaterBlock(state_dim=obs_feat_dim, obs_feat_dim=state_dim, **block_kwargs) 
            for _ in range(depth)
        ])

        # 为两个流都准备最终的 LayerNorm
        self.final_norm_state = nn.LayerNorm(state_dim)
        self.final_norm_obs = nn.LayerNorm(obs_feat_dim)

    def forward(self, state: torch.Tensor, obs_feat: torch.Tensor):
        """
        并行地更新 state 和 obs_feat。
        在每一层，两个流都使用上一层的输出进行交叉注意力计算。
        """
        all_cross_attn_weights = []
        
        # 使用 zip 并行遍历两个流的 blocks
        for state_block, obs_block in zip(self.state_blocks, self.obs_blocks):
            
            # --- 并行计算 ---
            # 1. 计算新的 state，此时它 "看" 的是上一层的 obs_feat
            new_state, state_cross_attn = state_block(state, obs_feat)
            
            # 2. 计算新的 obs_feat，此时它 "看" 的是上一层的 state
            # 注意输入的顺序是反的，以匹配 obs_block 的期望
            new_obs, obs_cross_attn = obs_block(obs_feat, state)
            
            # --- 更新状态以供下一层使用 ---
            state = new_state
            obs_feat = new_obs
            
            # 存储这一层两个方向的注意力图
            all_cross_attn_weights.append(
               state_cross_attn
            )
        
        # 在所有层处理完毕后，进行最终的归一化
        final_state = self.final_norm_state(state)
        final_obs = self.final_norm_obs(obs_feat)
        
        # 返回最后一层的 state, obs_feat 和所有层的注意力权重
        return final_state, final_obs, all_cross_attn_weights
        

class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            img_size=(240, 320),
            crop_shape=(76, 76),
            patch_size=16,
            embed_dim=768, # ViT patch embedding dimension
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            cond_dim=256,
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            state_dim=256,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        agent_pos_dim = shape_meta['obs']['agent_pos']['shape'][0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        self.state_dim = state_dim

        # self.obs_encoder = VisionTransformerEncoder(
        #     n_obs_steps=n_obs_steps,
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     embed_dim=embed_dim,
        #     agent_pos_dim=agent_pos_dim,
        #     out_dim=cond_dim # 编码器直接输出最终的条件维度
        # )
        self.num_patches = 30
        self.initial_state_tokens = nn.Parameter(
            torch.randn(1, self.num_patches, self.state_dim) * 0.02
        )
        
        # 为 state 序列创建一套独立的可学习位置编码
        self.state_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.state_dim)
        )
        self.state_updater = ParallelStateUpdater(
            depth=4, # 从 config 获取，例如 2 或 4
            state_dim=self.state_dim,
            obs_feat_dim=cond_dim,
            n_head=8,
            mlp_ratio=4,
            dropout=0
        )
        
        # if obs_encoder_group_norm:
        #     # replace batch norm with group norm
        #     replace_submodules(
        #         root_module=obs_encoder,
        #         predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        #         func=lambda x: nn.GroupNorm(
        #             num_groups=x.num_features//16, 
        #             num_channels=x.num_features)
        #     )
        #     # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # # obs_encoder.obs_randomizers['agentview_image']
        # if eval_fixed_crop:
        #     replace_submodules(
        #         root_module=obs_encoder,
        #         predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
        #         func=lambda x: dmvc.CropRandomizer(
        #             input_shape=x.input_shape,
        #             crop_height=x.crop_height,
        #             crop_width=x.crop_width,
        #             num_crops=x.num_crops,
        #             pos_enc=x.pos_enc
        #         )
        #     )

        # create diffusion model
        obs_feature_dim = 256
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

      
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.obs_projection_layer = nn.Linear(78, 256)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, 
                       obs_dict: Dict[str, torch.Tensor], 
                       state: Optional[torch.Tensor] = None # state 是不带位置编码的纯内容
                       ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        nobs_features = self.obs_projection_layer(nobs_features)
        
        # reshape back to B, To, Do
        
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        is_first_step = (state is None)
        if is_first_step:
            state_content = self.initial_state_tokens.expand(B, -1, -1)
        else:
            state_content = state
        
        state_with_pos = state_content + self.state_pos_embed
        new_state_with_pos,img, all_cross_attn = self.state_updater(
            state_with_pos, 
            nobs_features,
        )
        
        if is_first_step:
            update_gate = torch.ones(B, 1, 1, device=self.device)
        else:
            stacked_attn = torch.stack(all_cross_attn, dim=0)
            rearranged_attn = rearrange(stacked_attn, 'l b h n_state n_obs -> b n_state n_obs (l h)')
            state_query_img_key = rearranged_attn.mean(dim=(-1, -2))
            update_gate = torch.sigmoid(state_query_img_key).unsqueeze(-1)

        final_updated_state_with_pos = new_state_with_pos * update_gate + state_with_pos * (1 - update_gate)
        cond = img

        updated_state_content = final_updated_state_with_pos - self.state_pos_embed
            

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred, 
            'state': updated_state_content
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, state):
        # normalize input
        assert 'valid_mask' not in batch
        is_reset = batch.get('is_reset')
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = nactions.shape
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        nobs_features = self.obs_projection_layer(nobs_features)

        # reshape back to B, T, Do
            
        
        
        
        initial_state = self.initial_state_tokens.expand(B, -1, -1)


        old_state = initial_state if state is None else state


        state = torch.where(is_reset.view(B, 1, 1), initial_state, old_state)
        state_with_pos = state + self.state_pos_embed
        updated_state_with_pos,img, all_cross_attn = self.state_updater(state_with_pos, nobs_features)
        stacked_attn = torch.stack(all_cross_attn, dim=0)
        rearranged_attn = rearrange(stacked_attn, 'l b h n_state n_obs -> b n_state n_obs (l h)')
        state_query_img_key = rearranged_attn.mean(dim=(-1, -2))
        soft_gate = torch.sigmoid(state_query_img_key).unsqueeze(-1)
        calculated_gate = soft_gate
        update_gate = torch.where(is_reset.view(B, 1, 1), 1.0, calculated_gate)


        state_with_pos = updated_state_with_pos * update_gate + state_with_pos * (1 - update_gate)    
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, img)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        state_with_pos = state_with_pos - self.state_pos_embed
        return loss, state_with_pos
