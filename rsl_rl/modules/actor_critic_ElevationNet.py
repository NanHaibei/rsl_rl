# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, create_r2plus1d_feature_extractor, create_transformer_fusion_actor



class ActorCriticElevationNet(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # R(2+1)D 视觉编码器配置
        vision_input_channels: int = 1,
        vision_feature_dim: int = 64,
        vision_num_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (11, 11),
        # 本体信息编码器配置
        proprio_feature_dim: int = 128,
        # Transformer Fusion 配置
        transformer_hidden_dim: int = 256,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2,
        transformer_mlp_hidden_dims: list[int] | None = None,
        transformer_dropout: float = 0.1,
        transformer_use_proprio_embedding: bool = True,
        transformer_use_vision_embedding: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size  # 保存用于reshape
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            # 深度图是4D张量 [batch, frames, height, width]，跳过维度累加
            if obs_group == "height_scan_history":
                continue
            assert len(obs[obs_group].shape) == 2, f"The observation '{obs_group}' must be 1D (got shape {obs[obs_group].shape})."
            num_actor_obs += obs[obs_group].shape[-1]
        
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            # 深度图是4D张量，跳过维度累加
            if obs_group == "height_scan_history":
                continue
            assert len(obs[obs_group].shape) == 2, f"The observation '{obs_group}' must be 1D (got shape {obs[obs_group].shape})."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        ########################################## Actor ##############################################
        # 验证height_scan_history存在
        has_depth_input = False
        for group_name, group_keys in obs_groups.items():
            if "height_scan_history" in group_keys:
                has_depth_input = True
                break
        # 如果没有在obs_groups中找到，检查是否直接在obs中
        if not has_depth_input:
            has_depth_input = "height_scan_history" in obs.sorted_keys
        
        if not has_depth_input:
            raise ValueError(
                "ActorCriticElevationNet requires 'height_scan_history' in observations. "
                "Please add 'height_scan_history' to one of the obs_groups "
                "(e.g., obs_groups['policy'] or obs_groups['perception']). "
                "If you don't need vision input, please use RslRlPpoActorCriticCfg instead."
            )
        
        # 验证配置
        if True:  # 始终使用Transformer Fusion架构
            # 1. 验证height_scan_history的形状
            if "height_scan_history" not in obs:
                raise ValueError(
                    "height_scan_history is configured in obs_groups but not found in obs. "
                    "Please check your environment observation configuration."
                )
            
            # 提取实际的深度张量
            # height_scan_history可能是嵌套的TensorDict，需要递归提取
            depth_obs = obs["height_scan_history"]
            while isinstance(depth_obs, TensorDict):
                # 一直解包直到获得真正的Tensor
                keys = list(depth_obs.keys())
                if not keys:
                    raise ValueError("height_scan_history is an empty TensorDict")
                depth_obs = depth_obs[keys[0]]
            
            # 验证形状：固定为 [batch, frames, height*width]
            if len(depth_obs.shape) != 3:
                raise ValueError(
                    f"height_scan_history must be 3D tensor [batch, frames, height*width], "
                    f"but got shape {depth_obs.shape}"
                )
            
            batch_size, actual_frames, flattened_size = depth_obs.shape
            expected_height, expected_width = vision_spatial_size
            expected_size = expected_height * expected_width
            
            # 验证展开大小
            if flattened_size != expected_size:
                raise ValueError(
                    f"height_scan_history has flattened spatial size {flattened_size}, "
                    f"but expected {expected_size} (from vision_spatial_size {vision_spatial_size}). "
                    f"Please check your configuration."
                )
            
            # 2. 验证帧数
            if actual_frames != vision_num_frames:
                raise ValueError(
                    f"height_scan_history has {actual_frames} frames, "
                    f"but vision_num_frames is configured as {vision_num_frames}. "
                    f"Please adjust vision_num_frames in your config to match."
                )
            
            # 4. 验证Transformer参数
            if transformer_hidden_dim % transformer_num_heads != 0:
                raise ValueError(
                    f"transformer_hidden_dim ({transformer_hidden_dim}) must be divisible by "
                    f"transformer_num_heads ({transformer_num_heads})"
                )
            
            # 5. 不支持state_dependent_std
            if state_dependent_std:
                raise NotImplementedError(
                    "state_dependent_std is not supported with Transformer Fusion architecture. "
                    "Please set state_dependent_std=False in your config."
                )
            
            print("\n" + "="*80)
            print("🌟 Using Transformer Fusion Architecture with Depth Input")
            print("="*80)
            print(f"✓ Validated height_scan_history: [batch, {actual_frames}, {expected_height}×{expected_width}={expected_size}] (will reshape to [batch, {actual_frames}, {expected_height}, {expected_width}])")
            
            # 1. 本体信息处理流 - MLP特征提取器
            self.proprio_encoder = MLP(num_actor_obs, proprio_feature_dim, actor_hidden_dims, activation)
            print(f"✓ Proprioception Encoder MLP: {num_actor_obs} -> {proprio_feature_dim}")
            
            # 2. 深度图序列处理流 - R(2+1)D特征提取器
            self.elevation_net = create_r2plus1d_feature_extractor(
                input_channels=vision_input_channels,
                output_dim=vision_feature_dim,
                num_frames=vision_num_frames,
                spatial_size=vision_spatial_size
            )
            print(f"✓ Vision Encoder R(2+1)D: [{vision_input_channels}, {vision_num_frames}, {vision_spatial_size[0]}, {vision_spatial_size[1]}] -> {vision_feature_dim}")
            
            # 3. Transformer Fusion模块 + MLP映射到动作
            if transformer_mlp_hidden_dims is None:
                transformer_mlp_hidden_dims = [256, 128]
            
            self.fusion_actor = create_transformer_fusion_actor(
                proprioception_dim=proprio_feature_dim,
                vision_feature_dim=vision_feature_dim,
                num_actions=num_actions,
                hidden_dim=transformer_hidden_dim,
                num_heads=transformer_num_heads,
                num_layers=transformer_num_layers,
                mlp_hidden_dims=transformer_mlp_hidden_dims,
                dropout=transformer_dropout,
                use_proprio_embedding=transformer_use_proprio_embedding,
                use_vision_embedding=transformer_use_vision_embedding,
            )
            print("="*80 + "\n")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()


        ########################################## Critic ##############################################
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        # Transformer Fusion架构不支持state_dependent_std（已在前面验证）
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict, proprio_obs: torch.Tensor) -> None:
        """更新动作分布
        
        Args:
            obs: 原始观测TensorDict（包含深度图等）
            proprio_obs: 已处理的本体观测（已归一化）
        """
        # 1. 提取本体特征
        proprio_features = self.proprio_encoder(proprio_obs)  # [batch, 128]
        
        # 2. 提取深度图并处理
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            # 一直解包直到获得真正的Tensor
            keys = list(depth_obs.keys())
            if not keys:
                raise ValueError("height_scan_history is an empty TensorDict")
            depth_obs = depth_obs[keys[0]]
        
        # 验证解包后的形状
        if not isinstance(depth_obs, torch.Tensor):
            raise TypeError(f"After unpacking, depth_obs should be a Tensor, but got {type(depth_obs)}")
        
        if len(depth_obs.shape) != 3:
            raise ValueError(
                f"depth_obs should have 3 dimensions [batch, frames, height*width], "
                f"but got shape {depth_obs.shape}"
            )
        
        # Reshape: [batch, frames, height*width] -> [batch, frames, height, width]
        batch_size, num_frames, _ = depth_obs.shape
        expected_height, expected_width = self.vision_spatial_size
        depth_obs = depth_obs.view(batch_size, num_frames, expected_height, expected_width)
        
        vision_features = self.elevation_net(depth_obs)  # [batch, 64]
        
        # 3. Transformer融合并生成动作均值
        mean = self.fusion_actor(proprio_features, vision_features)  # [batch, num_actions]
        
        # 4. 计算标准差
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        self._update_distribution(obs, proprio_obs)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # Transformer Fusion架构
        proprio_features = self.proprio_encoder(proprio_obs)
        
        # 提取深度图并处理
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            # 一直解包直到获得真正的Tensor
            keys = list(depth_obs.keys())
            if not keys:
                raise ValueError("height_scan_history is an empty TensorDict")
            depth_obs = depth_obs[keys[0]]
        
        # 验证解包后的形状
        if not isinstance(depth_obs, torch.Tensor):
            raise TypeError(f"After unpacking, depth_obs should be a Tensor, but got {type(depth_obs)}")
        
        if len(depth_obs.shape) != 3:
            raise ValueError(
                f"depth_obs should have 3 dimensions [batch, frames, height*width], "
                f"but got shape {depth_obs.shape}"
            )
        
        # Reshape: [batch, frames, height*width] -> [batch, frames, height, width]
        batch_size, num_frames, _ = depth_obs.shape
        expected_height, expected_width = self.vision_spatial_size
        depth_obs = depth_obs.view(batch_size, num_frames, expected_height, expected_width)
        
        vision_features = self.elevation_net(depth_obs)
        mean = self.fusion_actor(proprio_features, vision_features)
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取actor的本体观测（排除深度图）"""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            # 深度图单独处理，不加入本体观测
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic观测（排除深度图）"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            # 深度图不用于critic（可以根据需求修改）
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["critic"][0]].shape[0], 0)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

