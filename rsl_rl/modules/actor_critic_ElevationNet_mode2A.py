# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode2A: 简化融合架构

网络结构:
    本体观测 -> 直接进入融合层
    高程图 -> 高程图编码器MLP -> 特征 -> 融合层
    [本体观测 + 特征] -> 融合MLP -> 动作

与Mode2的区别:
- 去掉本体观测的特征提取网络
- 本体观测值直接进入fusion层
- 高程图先进MLP提取特征再进fusion层
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
import copy
import os

class ActorCriticElevationNetMode2A(nn.Module):
    """Mode2A: 本体观测直接融合，高程图特征提取后融合
    
    三个MLP网络:
    1. 高程图编码器MLP - 提取视觉特征
    2. Actor融合MLP - 融合本体观测和视觉特征后输出动作
    3. Critic MLP - 价值评估
    """
    
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg=None,
        alg_cfg: dict | None = None,
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
        print("🌟 网络架构: ElevationNet Mode2A")
        print("=" * 80)
        print("✓ Mode2A: 本体观测直接融合 + 高程图编码器 + 融合网络 -> 动作")
        
        # 1. 高程图编码器MLP
        self.elevation_net = MLP(height_map_input_dim, vision_feature_dim, elevation_encoder_hidden_dims, "elu")
        print(f"  1. 高程图编码器: {height_map_input_dim} ({vision_num_frames}×{height_map_dim}) -> {elevation_encoder_hidden_dims} -> {vision_feature_dim}")
        
        # 2. Actor融合MLP
        fusion_input_dim = num_actor_obs + vision_feature_dim
        self.fusion_actor = MLP(fusion_input_dim, num_actions, fusion_actor_hidden_dims, "elu")
        print(f"  2. 融合MLP: {fusion_input_dim} (本体观测{num_actor_obs} + 视觉特征{vision_feature_dim}) -> {fusion_actor_hidden_dims} -> {num_actions}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        ########################################## Critic ##############################################
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"  3. Critic MLP: {num_critic_obs} -> {critic_hidden_dims} -> 1")
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

    # def _extract_height_map_sequence(self, obs: TensorDict) -> torch.Tensor:
    #     """提取高程图序列并展平"""
    #     depth_obs = obs["height_scan_history"]
    #     while isinstance(depth_obs, TensorDict):
    #         keys = list(depth_obs.keys())
    #         depth_obs = depth_obs[keys[0]]
    #     # 展平所有帧: [batch, frames, height*width] -> [batch, frames*height*width]
    #     batch_size = depth_obs.shape[0]
    #     return depth_obs.view(batch_size, -1)

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
        
        # 2. 提取高程图特征
        # height_map_sequence = self._extract_height_map_sequence(obs)
        height_map_sequence = obs["height_scan_history"].view(obs["height_scan_history"].shape[0], -1)
        vision_features = self.elevation_net(height_map_sequence)
        
        # 3. 融合本体观测和视觉特征并输出动作
        fused_features = torch.cat([proprio_obs, vision_features], dim=-1)
        mean = self.fusion_actor(fused_features)
        
        self._update_distribution(mean)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 获取并归一化本体观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 提取高程图特征
        # height_map_sequence = self._extract_height_map_sequence(obs)
        height_map_sequence = obs["height_scan_history"].view(obs["height_scan_history"].shape[0], -1)
        vision_features = self.elevation_net(height_map_sequence)
        
        # 3. 融合本体观测和视觉特征并输出动作（确定性）
        fused_features = torch.cat([proprio_obs, vision_features], dim=-1)
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

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典，包含主要的优化器
        """
        import torch.optim as optim
        
        optimizer = optim.Adam([
            {'params': self.elevation_net.parameters()},
            {'params': self.fusion_actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer
        }

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode2A_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode2A策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode2A_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode2A专用的导出器
        exporter = _ElevationNetMode2AOnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode2AOnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode2A策略的ONNX导出器"""

    def __init__(self, policy: ActorCriticElevationNetMode2A, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 复制策略参数
        if hasattr(policy, "elevation_net"):
            self.elevation_net = copy.deepcopy(policy.elevation_net)
        if hasattr(policy, "fusion_actor"):
            self.fusion_actor = copy.deepcopy(policy.fusion_actor)

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        # 假设输入是 [本体观测 + 高程图序列展平]
        # 需要分离本体观测和高程图
        obs_len = self.normalizer.in_features
        proprio_obs = x[:, 0:obs_len]
        height_map_sequence = x[:, obs_len:]
        
        # 归一化本体观测
        normalized_obs = self.normalizer(proprio_obs)
        
        # 提取高程图特征
        vision_features = self.elevation_net(height_map_sequence)
        
        # 融合本体观测和视觉特征并输出动作
        fused_features = torch.cat([normalized_obs, vision_features], dim=-1)
        actions_mean = self.fusion_actor(fused_features)
        return actions_mean

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        # 创建输入示例
        total_dim = self.normalizer.in_features + self.elevation_net[0].in_features
        obs = torch.zeros(1, total_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
