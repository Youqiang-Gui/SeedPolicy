from typing import Dict, Tuple, Union, Optional
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
    ):
        """
        Modified version:
        1. Forces Resize(76)
        2. Forces Fusion MLP to output 512 dims
        3. Fuses Image Feats + LowDim Feats (Agent Pos)
        """
        super().__init__()

        # ================= 配置区域 (硬编码设置) =================
        # 强制使用 Resize 76 (短边模式)
        self.force_resize = 76 
        # 强制输出 512 维 (适合你的 CNN Policy)
        self.fusion_output_dim = 512
        # =======================================================

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # 处理 Backbone 共享
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            
            if type == "rgb":
                rgb_keys.append(key)
                # 配置模型
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    # 自动替换 BatchNorm 为 GroupNorm (无需外部依赖)
                    if use_group_norm:
                        self._replace_bn_with_gn(this_model)
                    key_model_map[key] = this_model

                # --- 核心修改：强制 Resize ---
                # 无论 YAML 传什么，都使用 Resize(76, antialias=True)
                this_resizer = T.Resize(size=self.force_resize, antialias=True)
                
                # 配置 Crop (如果需要的话，通常 Resize 后不需要 RandomCrop，除非为了增强)
                # 这里保留原有逻辑，但在 76 模式下通常 crop_shape 会设为 null
                input_shape = (shape[0], self.force_resize, self.force_resize) # 近似 shape
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                # 配置 Normalize
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        # --- 核心修改：自动构建 Fusion MLP (输出 512) ---
        # 1. 计算输入维度 (Image Features + Low Dim Features)
        with torch.no_grad():
            dummy_dict = {}
            for key, attr in obs_shape_meta.items():
                shape = tuple(attr["shape"])
                dummy_dict[key] = torch.zeros((1,) + shape)
            
            # 跑一次前向传播来确定维度
            raw_out = self._raw_forward(dummy_dict)
            input_dim = raw_out.shape[-1]
            print(f"[AutoEncoder] Detected Input Dim: {input_dim} | Target Output Dim: {self.fusion_output_dim}")

        # 2. 定义 MLP (ResNet+State -> 512 -> 512)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.fusion_output_dim),
            nn.LayerNorm(self.fusion_output_dim)
        )

    def _replace_bn_with_gn(self, m):
        """递归替换 BN 为 GN"""
        for name, child in m.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = 32
                num_channels = child.num_features
                while num_channels % num_groups != 0 and num_groups > 1:
                    num_groups //= 2
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                setattr(m, name, gn)
            else:
                self._replace_bn_with_gn(child)

    def _raw_forward(self, obs_dict):
        """仅提取特征并拼接，不经过 MLP"""
        batch_size = None
        features = list()
        
        # 处理 RGB
        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # 处理 Low Dim (State)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)

        # 拼接
        result = torch.cat(features, dim=-1)
        return result

    def forward(self, obs_dict):
        # 1. 提取所有特征 (结果维度可能是 512 + 14 = 526)
        raw_result = self._raw_forward(obs_dict)
        
        # 2. 通过 MLP 融合 (结果固定为 512)
        result = self.fusion_mlp(raw_result)
        
        return result

    @torch.no_grad()
    def output_shape(self):
        # 返回固定的 output dim
        return (self.fusion_output_dim,)