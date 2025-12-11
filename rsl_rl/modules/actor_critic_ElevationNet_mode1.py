# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode1: 最简单的高程图网络架构

网络结构: [本体观测 + 高程图] -> MLP -> 动作
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticElevationNetMode1(nn.Module):
    """Mode1: 本体观测和高程图拼接后直接进MLP输出动作
    
    这是最简单的架构，参数最少：
    - 将本体观测和高程图展平后直接拼接
    - 通过单个MLP直接输出动作
    - 适合快速实验和baseline对比
    """
    
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [768, 384, 128],
        critic_hidden_dims: tuple[int] | list[int] = [768, 384, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        vision_num_frames: int = 1,
        vision_spatial_size: tuple[int, int] = (25, 17),
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
        
        ########################################## Actor ##############################################
        print("\n" + "=" * 80)
        print("🌟 网络架构: ElevationNet Mode1")
        print("=" * 80)
        print("✓ Mode1: [本体观测 + 高程图] -> MLP -> 动作")
        
        # 直接Actor: 拼接本体观测和高程图后输出动作
        input_dim = num_actor_obs + height_map_dim
        self.direct_actor = MLP(input_dim, num_actions, actor_hidden_dims, activation)
        print(f"  Actor MLP: {input_dim} (本体{num_actor_obs} + 高程图{height_map_dim}) -> {actor_hidden_dims} -> {num_actions}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        ########################################## Critic ##############################################
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"  Critic MLP: {num_critic_obs} -> {critic_hidden_dims} -> 1")
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

    def _extract_height_map(self, obs: TensorDict) -> torch.Tensor:
        """提取高程图的最新一帧展平数据"""
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            keys = list(depth_obs.keys())
            depth_obs = depth_obs[keys[0]]
        # 取最新一帧: [batch, frames, height*width] -> [batch, height*width]
        return depth_obs[:, -1, :]

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 拼接本体观测和高程图单帧
        height_map = self._extract_height_map(obs)
        actor_input = torch.cat([proprio_obs, height_map], dim=-1)
        
        # 通过MLP输出动作均值
        mean = self.direct_actor(actor_input)
        self._update_distribution(mean)
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 拼接本体观测和高程图
        height_map = self._extract_height_map(obs)
        actor_input = torch.cat([proprio_obs, height_map], dim=-1)
        
        # 返回动作均值（确定性）
        mean = self.direct_actor(actor_input)
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
