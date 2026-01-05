# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode9: 简化架构，2D CNN处理高程图序列 + 直接特征融合

网络结构:
    高程图历史(50帧) -> 采样(隔10帧抽1帧) -> 5帧序列 -> 2D CNN -> 64维特征向量
    本体观测(当前帧131维) -> 直接使用
    [64维视觉特征 + 131维本体观测] -> 195维融合特征 -> Actor/Critic MLP
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


class Conv2DElevationEncoder(nn.Module):
    """2D CNN编码器，用于处理采样后的高程图序列
    
    将5帧高程图作为5个通道处理，类似于多通道图像
    """
    
    def __init__(self, 
                 num_frames=5,
                 hidden_dims=[16, 32, 64],
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 out_dim=64,
                 vision_spatial_size=(25, 17)):
        super().__init__()
        
        # 动态构建卷积层
        layers = []
        in_channels = num_frames
        
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过CNN后的特征维度（通过前向传播计算）
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_frames, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def forward(self, x):
        # x: [B, 5, 25, 17]
        x_mean = x.mean(dim = (-1, -2), keepdim=True)
        x = torch.clip((x - x_mean) / 0.1, -5.0, 5.0) # 简陋的归一化
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ActorCriticElevationNetMode9(nn.Module):
    """Mode9: 简化架构，2D CNN处理高程图序列 + 直接特征融合
    
    网络组成:
    1. 高程图采样器: 从50帧历史中隔10帧抽取5帧
    2. 2D CNN编码器: 将5帧作为5个通道处理，输出64维特征
    3. 特征融合: 拼接64维视觉特征和131维本体观测成195维
    4. Actor MLP: 从融合特征输出动作
    5. Critic MLP: 从融合特征输出价值
    """
    
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg=None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # 高程图编码器配置
        elevation_sampled_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (25, 17),
        vision_feature_dim: int = 64,
        # 2D CNN配置
        conv2d_hidden_dims: list[int] = [16, 32, 64],
        conv2d_kernel_sizes: list[int] = [3, 3, 3],
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        # 配置
        self.cfg = kwargs
        self.extra_info = dict()
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size
        self.noise_std_type = noise_std_type
        self.elevation_sampled_frames = elevation_sampled_frames
        self.vision_feature_dim = vision_feature_dim
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        
        ########################################## 网络架构 ##############################################
        
        # 1. Actor网络
        self.elevation_encoder_actor = Conv2DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.actor = MLP(num_actor_obs + vision_feature_dim, num_actions, actor_hidden_dims, activation)
        
        # 2. Critic网络
        self.elevation_encoder_critic = Conv2DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.critic = MLP(num_critic_obs + vision_feature_dim, 1, critic_hidden_dims, activation)

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

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
        
        # 打印网络结构
        print("\n" + "="*80)
        print("ActorCriticElevationNetMode9 网络结构")
        print("="*80)
        print(self)
        print("="*80 + "\n")

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

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        current_proprio_obs = obs["policy"]
        
        # 应用观测归一化
        current_proprio_obs = self.actor_obs_normalizer(current_proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取高程图特征
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 4. 融合特征
        fused_features = torch.cat((vision_features, current_proprio_obs), dim=-1)
        
        # 5. Actor输出动作
        mean = self.actor(fused_features)
        self._update_distribution(mean)
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        current_proprio_obs = obs["policy"]
        
        # 应用观测归一化
        current_proprio_obs = self.actor_obs_normalizer(current_proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取高程图特征
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 4. 融合特征
        fused_features = torch.cat((vision_features, current_proprio_obs), dim=-1)
        
        # 5. Actor输出动作
        mean = self.actor(fused_features)
        
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值"""
        # 1. 提取观测值
        height_maps = obs["height_scan_critic"]
        current_proprio_obs = obs["critic"]
        
        # 应用观测归一化
        current_proprio_obs = self.critic_obs_normalizer(current_proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取高程图特征
        vision_features = self.elevation_encoder_critic(sampled_height_maps)
        
        # 4. 融合特征
        fused_features = torch.cat((vision_features, current_proprio_obs), dim=-1)
        
        # 5. Critic输出价值
        value = self.critic(fused_features)
        
        return value

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取actor的本体观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
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

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典，包含主要的优化器
        """
        import torch.optim as optim
        
        # Mode9没有编码器优化器，所有参数一起优化
        optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.elevation_encoder_actor.parameters()},
            {'params': self.elevation_encoder_critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode9_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode9策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode9_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode9专用的导出器
        exporter = _ElevationNetMode9OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode9OnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode9策略的ONNX导出器"""

    def __init__(self, policy: ActorCriticElevationNetMode9, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 复制策略参数
        if hasattr(policy, "elevation_encoder_actor"):
            self.elevation_encoder = copy.deepcopy(policy.elevation_encoder_actor)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        self.elevation_sampled_frames = policy.elevation_sampled_frames
        self.vision_spatial_size = policy.vision_spatial_size
        # 从actor网络计算本体观测维度：actor输入维度 - 视觉特征维度
        actor_input_dim = policy.actor[0].in_features  # MLP第一层的输入维度
        self.proprio_obs_dim = actor_input_dim - policy.vision_feature_dim

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_input):
        """前向传播（单输入版本）
        
        Args:
            obs_input: 合并的观测数据，形状为 [batch_size, total_obs_dim]
                       前 proprio_obs_dim 维是本体观测
                       后面是展平的高程图数据（需要reshape为 [B, sampled_frames, height, width]）
        
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_input.shape[0]
        
        # 切片分离本体观测和高程图数据
        proprio_data = obs_input[:, :self.proprio_obs_dim]
        elevation_data_flat = obs_input[:, self.proprio_obs_dim:]
        
        # 将高程图数据reshape为 [B, sampled_frames, height, width]
        height, width = self.vision_spatial_size
        elevation_data = elevation_data_flat.reshape(
            batch_size, self.elevation_sampled_frames, height, width
        )
        
        # 应用归一化器到本体观测（与act函数一致）
        proprio_data = self.normalizer(proprio_data)
        
        # 提取视觉特征
        vision_features = self.elevation_encoder(elevation_data)
        
        # 融合特征
        fused_features = torch.cat([vision_features, proprio_data], dim=-1)
        
        # 输出动作
        actions_mean = self.actor(fused_features)
        return actions_mean

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        
        # 计算维度
        height, width = self.vision_spatial_size
        sampled_frames = self.elevation_sampled_frames
        elevation_dim = sampled_frames * height * width
        total_obs_dim = self.proprio_obs_dim + elevation_dim
        
        # 创建单个合并的输入示例
        # 格式: [本体观测(102维) + 展平的高程图(5*25*17=2125维)] = 2227维
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (单输入模式):")
        print(f"{'='*80}")
        print(f"  本体观测维度:     {self.proprio_obs_dim}")
        print(f"  高程图维度:       {elevation_dim} ({sampled_frames}×{height}×{width})")
        print(f"  总输入维度:       {total_obs_dim}")
        print(f"  输入切片方式:     [:, :{self.proprio_obs_dim}] = 本体, "
              f"[:, {self.proprio_obs_dim}:] = 高程图")
        print(f"{'='*80}\n")
        
        torch.onnx.export(
            self,
            obs_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
