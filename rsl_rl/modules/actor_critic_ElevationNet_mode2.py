# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode2: 特征提取+融合架构

网络结构:
    本体 -> 本体编码器MLP -> 特征1
    高程图 -> 高程图编码器MLP -> 特征2
    [特征1 + 特征2] -> 融合MLP -> 动作
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticElevationNetMode2(nn.Module):
    """Mode2: 本体和高程图分别提取特征后融合输出动作
    
    四个MLP网络:
    1. 本体编码器MLP - 提取本体特征
    2. 高程图编码器MLP - 提取视觉特征
    3. Actor融合MLP - 融合特征后输出动作
    4. Critic MLP - 价值评估
    """
    
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # 高程图编码器配置
        vision_feature_dim: int = 64,
        vision_num_frames: int = 1,
        vision_spatial_size: tuple[int, int] = (25, 17),
        elevation_encoder_hidden_dims: list[int] | None = None,
        # 本体编码器配置
        proprio_feature_dim: int = 64,
        proprio_encoder_hidden_dims: list[int] | None = None,
        # 融合网络配置
        fusion_actor_hidden_dims: list[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        # 配置
        self.cfg = kwargs
        self.extra_info = dict()
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size
        self.noise_std_type = noise_std_type
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"] if g != "height_scan_history")
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"] if g != "height_scan_history")
        
        # 计算高程图展平后的维度
        height, width = vision_spatial_size
        height_map_dim = height * width
        height_map_input_dim = vision_num_frames * height_map_dim
        
        ########################################## Actor ##############################################
        print("\n" + "=" * 80)
        print("🌟 网络架构: ElevationNet Mode2")
        print("=" * 80)
        print("✓ Mode2: 本体编码器 + 高程图编码器 + 融合网络 -> 动作")
        
        # 1. 本体编码器MLP
        if proprio_encoder_hidden_dims is None:
            proprio_encoder_hidden_dims = actor_hidden_dims
        self.proprio_encoder = MLP(num_actor_obs, proprio_feature_dim, proprio_encoder_hidden_dims, activation)
        print(f"  1. 本体编码器: {num_actor_obs} -> {proprio_encoder_hidden_dims} -> {proprio_feature_dim}")
        
        # 2. 高程图编码器MLP
        if elevation_encoder_hidden_dims is None:
            elevation_encoder_hidden_dims = [max(height_map_input_dim // 2, vision_feature_dim * 2), vision_feature_dim * 2]
        self.elevation_net = MLP(height_map_input_dim, vision_feature_dim, elevation_encoder_hidden_dims, "elu")
        print(f"  2. 高程图编码器: {height_map_input_dim} ({vision_num_frames}×{height_map_dim}) -> {elevation_encoder_hidden_dims} -> {vision_feature_dim}")
        
        # 3. Actor融合MLP
        if fusion_actor_hidden_dims is None:
            fusion_actor_hidden_dims = actor_hidden_dims if actor_hidden_dims else [256, 128]
            print(f"     ℹ️  fusion_actor_hidden_dims未设置，使用actor_hidden_dims={fusion_actor_hidden_dims}")
        fusion_input_dim = proprio_feature_dim + vision_feature_dim
        self.fusion_actor = MLP(fusion_input_dim, num_actions, fusion_actor_hidden_dims, "elu")
        print(f"  3. 融合MLP: {fusion_input_dim} (本体{proprio_feature_dim} + 视觉{vision_feature_dim}) -> {fusion_actor_hidden_dims} -> {num_actions}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        ########################################## Critic ##############################################
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"  4. Critic MLP: {num_critic_obs} -> {critic_hidden_dims} -> 1")
        print("=" * 80 + "\n")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        ########################################## Action Noise ##############################################
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        # Action distribution
        self.distribution = None
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

    def _extract_height_map_sequence(self, obs: TensorDict) -> torch.Tensor:
        """提取高程图序列并展平"""
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            keys = list(depth_obs.keys())
            depth_obs = depth_obs[keys[0]]
        # 展平所有帧: [batch, frames, height*width] -> [batch, frames*height*width]
        batch_size = depth_obs.shape[0]
        return depth_obs.view(batch_size, -1)

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 获取并归一化本体观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 提取本体特征
        proprio_features = self.proprio_encoder(proprio_obs)
        
        # 3. 提取高程图特征
        height_map_sequence = self._extract_height_map_sequence(obs)
        vision_features = self.elevation_net(height_map_sequence)
        
        # 4. 融合特征并输出动作
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        mean = self.fusion_actor(fused_features)
        
        self._update_distribution(mean)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 获取并归一化本体观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 提取本体特征
        proprio_features = self.proprio_encoder(proprio_obs)
        
        # 3. 提取高程图特征
        height_map_sequence = self._extract_height_map_sequence(obs)
        vision_features = self.elevation_net(height_map_sequence)
        
        # 4. 融合特征并输出动作（确定性）
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        mean = self.fusion_actor(fused_features)
        
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值"""
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取actor的本体观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["critic"][0]].shape[0], 0)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新观测归一化统计量"""
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True
