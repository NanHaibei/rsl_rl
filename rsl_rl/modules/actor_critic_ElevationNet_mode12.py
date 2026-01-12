# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode12: R(2+1)D处理特权历史高程图 + MLP处理本体观测

网络结构:
    Critic Pipeline:
    - 单帧本体特权观测 -> MLP提取特征
    - 特权历史高程图序列 -> R(2+1)D提取特征
    - 两个特征向量 -> Critic网络 -> Value
    
    Actor Pipeline:
    - 单帧本体观测 -> MLP提取特征
    - 历史高程图序列 -> R(2+1)D提取特征
    - 两个特征向量 -> Actor网络 -> Actions
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


class R21DElevationEncoder(nn.Module):
    """R(2+1)D编码器，用于处理高程图序列
    
    R(2+1)D将3D卷积分解为：
    1. 2D空间卷积（处理每一帧的空间信息）
    2. 1D时间卷积（处理帧间的时序信息）
    
    这种分解可以减少参数量并提高性能
    """
    
    def __init__(self, 
                 num_frames=5,
                 hidden_dims=[16, 32, 64],
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 out_dim=64,
                 vision_spatial_size=(25, 17)):
        super().__init__()
        
        self.num_frames = num_frames
        
        # 构建R(2+1)D卷积层
        layers = []
        in_channels = 1  # 单通道高程图
        
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            # R(2+1)D块：空间卷积 + 时间卷积
            layers.append(self._make_r21d_block(
                in_channels, 
                hidden_dim, 
                kernel_size, 
                stride
            ))
            in_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过卷积后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_frames, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def _make_r21d_block(self, in_channels, out_channels, kernel_size, stride):
        """创建一个R(2+1)D块
        
        包含：
        1. 2D空间卷积：(1, kernel_size, kernel_size)
        2. 1D时间卷积：(kernel_size, 1, 1)
        """
        # 中间通道数（用于分解）
        mid_channels = (in_channels * out_channels * kernel_size * kernel_size) // (in_channels * kernel_size * kernel_size + out_channels)
        mid_channels = max(mid_channels, 1)  # 确保至少有1个通道
        
        # 空间卷积：处理每一帧的空间信息
        spatial_conv = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                mid_channels, 
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, kernel_size//2, kernel_size//2)
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 时间卷积：处理帧间的时序信息
        temporal_conv = nn.Sequential(
            nn.Conv3d(
                mid_channels, 
                out_channels, 
                kernel_size=(kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(kernel_size//2, 0, 0)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        return nn.Sequential(spatial_conv, temporal_conv)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W] 高程图序列
        Returns:
            features: [B, out_dim] 特征向量
        """
        # 添加通道维度: [B, T, H, W] -> [B, 1, T, H, W]
        x = x.unsqueeze(1)
        
        # 归一化
        x_mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        x = torch.clip((x - x_mean) / 0.6, -5.0, 5.0)
        
        # 通过R(2+1)D卷积
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ActorCriticElevationNetMode12(nn.Module):
    """Mode12: R(2+1)D处理特权历史高程图 + MLP处理本体观测
    
    网络组成:
    1. Actor部分:
       - MLP特征提取器: 处理本体观测 -> 特征向量
       - R(2+1)D编码器: 处理历史高程图 -> 视觉特征向量
       - Actor网络: [MLP特征 + 视觉特征] -> 动作
    
    2. Critic部分:
       - MLP特征提取器: 处理本体特权观测 -> 特征向量
       - R(2+1)D编码器: 处理特权历史高程图 -> 视觉特征向量
       - Critic网络: [MLP特征 + 视觉特征] -> 价值
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
        critic_hidden_dims: tuple[int] | list[int] = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # 高程图编码器配置
        elevation_sampled_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (25, 17),
        vision_feature_dim: int = 64,
        # R(2+1)D配置
        r21d_hidden_dims: list[int] = [16, 32, 64],
        r21d_kernel_sizes: list[int] = [3, 3, 3],
        # MLP特征提取器配置
        actor_mlp_feature_dim: int = 64,
        critic_mlp_feature_dim: int = 64,
        mlp_extractor_hidden_dims: tuple[int] | list[int] = [128],
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
        self.actor_mlp_feature_dim = actor_mlp_feature_dim
        self.critic_mlp_feature_dim = critic_mlp_feature_dim
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        
        ########################################## 网络架构 ##############################################
        
        # 1. Actor网络
        # MLP特征提取器：本体观测 -> 特征向量
        self.actor_mlp_extractor = MLP(
            num_actor_obs, 
            actor_mlp_feature_dim, 
            mlp_extractor_hidden_dims, 
            activation
        )
        
        # R(2+1)D编码器：高程图序列 -> 视觉特征向量
        self.elevation_encoder_actor = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # Actor主网络：融合特征 -> 动作
        self.actor = MLP(
            actor_mlp_feature_dim + vision_feature_dim, 
            num_actions, 
            actor_hidden_dims, 
            activation
        )
        
        # 2. Critic网络
        # MLP特征提取器：本体特权观测 -> 特征向量
        self.critic_mlp_extractor = MLP(
            num_critic_obs, 
            critic_mlp_feature_dim, 
            mlp_extractor_hidden_dims, 
            activation
        )
        
        # R(2+1)D编码器：特权高程图序列 -> 视觉特征向量
        self.elevation_encoder_critic = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # Critic主网络：融合特征 -> 价值
        self.critic = MLP(
            critic_mlp_feature_dim + vision_feature_dim, 
            1, 
            critic_hidden_dims, 
            activation
        )

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
        print("ActorCriticElevationNetMode12 网络结构")
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
        
        # 3. 提取MLP特征
        mlp_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 5. 融合特征
        fused_features = torch.cat((mlp_features, vision_features), dim=-1)
        
        # 6. Actor输出动作
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
        
        # 3. 提取MLP特征
        mlp_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 5. 融合特征
        fused_features = torch.cat((mlp_features, vision_features), dim=-1)
        
        # 6. Actor输出动作
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
        
        # 3. 提取MLP特征
        mlp_features = self.critic_mlp_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征
        vision_features = self.elevation_encoder_critic(sampled_height_maps)
        
        # 5. 融合特征
        fused_features = torch.cat((mlp_features, vision_features), dim=-1)
        
        # 6. Critic输出价值
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
        
        optimizer = optim.Adam([
            {'params': self.actor_mlp_extractor.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic_mlp_extractor.parameters()},
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

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode12L_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode12策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode12L_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode12专用的导出器
        exporter = _ElevationNetMode12OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode12OnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode12策略的ONNX导出器"""

    def __init__(self, policy: ActorCriticElevationNetMode12, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # 复制策略参数
        if hasattr(policy, "actor_mlp_extractor"):
            self.actor_mlp_extractor = copy.deepcopy(policy.actor_mlp_extractor)
        if hasattr(policy, "elevation_encoder_actor"):
            self.elevation_encoder = copy.deepcopy(policy.elevation_encoder_actor)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        self.elevation_sampled_frames = policy.elevation_sampled_frames
        self.vision_spatial_size = policy.vision_spatial_size
        
        # 从MLP特征提取器获取本体观测维度
        self.proprio_obs_dim = policy.actor_mlp_extractor[0].in_features
        self.mlp_feature_dim = policy.actor_mlp_feature_dim

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
        
        # 应用归一化器到本体观测
        proprio_data = self.normalizer(proprio_data)
        
        # 提取MLP特征
        mlp_features = self.actor_mlp_extractor(proprio_data)
        
        # 提取视觉特征
        vision_features = self.elevation_encoder(elevation_data)
        
        # 融合特征
        fused_features = torch.cat([mlp_features, vision_features], dim=-1)
        
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
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (单输入模式 - Mode12):")
        print(f"{'='*80}")
        print(f"  本体观测维度:     {self.proprio_obs_dim}")
        print(f"  MLP特征维度:      {self.mlp_feature_dim}")
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
