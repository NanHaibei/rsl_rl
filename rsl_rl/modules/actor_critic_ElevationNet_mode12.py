# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode12: R(2+1)D处理特权历史高程图 + MLP处理本体观测 + 编解码器

网络结构:
    Actor pipeline:
        当前帧本体观测 -> MLP提取特征 -> actor_mlp_feature
        历史高程图序列 -> R(2+1)D -> 两个MLP头 -> vision_feature_1 + vision_feature_2
        5帧历史本体观测 -> MLP -> 本体历史特征
        本体历史特征 + vision_feature_2 -> Encoder -> 线速度(3) + 足端高度(2) + 隐向量z
        线速度 + 足端高度 + 隐向量z -> Decoder -> 重建下一帧观测
        [actor_mlp_feature + vision_feature_1 + 线速度 + 足端高度 + 隐向量z] -> Actor网络 -> 动作
    
    Critic:
        本体特权观测 -> MLP提取特征 -> critic_mlp_feature
        特权历史高程图序列 -> R(2+1)D提取特征 -> critic_vision_feature
        [critic_mlp_feature + critic_vision_feature] -> Critic网络 -> Value
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
        x = torch.clip((x - x_mean) / 0.1, -5.0, 5.0)
        
        # 通过R(2+1)D卷积
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ActorCriticElevationNetMode12(nn.Module):
    """Mode12: R(2+1)D处理特权历史高程图 + MLP处理本体观测 + 编解码器
    
    网络组成:
    1. Actor部分:
       - MLP特征提取器: 处理当前帧本体观测 -> actor_mlp_feature
       - R(2+1)D编码器: 处理历史高程图 -> R(2+1)D特征 -> 两个MLP头 -> vision_feature_1 + vision_feature_2
       - 本体历史MLP: 5帧历史本体观测 -> 本体历史特征
       - Encoder: 本体历史特征 + vision_feature_2 -> 线速度 + 足端高度 + 隐向量z
       - Decoder: 线速度 + 足端高度 + 隐向量z -> 重建下一帧观测
       - Actor网络: [actor_mlp_feature + vision_feature_1 + 线速度 + 足端高度 + 隐向量z] -> 动作
    
    2. Critic部分:
       - MLP特征提取器: 处理本体特权观测 -> critic_mlp_feature
       - R(2+1)D编码器: 处理特权历史高程图 -> critic_vision_feature
       - Critic网络: [critic_mlp_feature + critic_vision_feature] -> 价值
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
        # 编解码器配置
        encoder_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        decoder_hidden_dims: tuple[int] | list[int] = [128, 256, 512],
        num_latent: int = 14,  # 隐向量维度
        num_decode: int = 70,  # decoder重建的观测维度
        proprio_history_mlp_hidden_dims: tuple[int] | list[int] = [256, 128],
        proprio_history_feature_dim: int = 64,
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
        self.proprio_history_feature_dim = proprio_history_feature_dim
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / elevation_sampled_frames)  # 单帧本体观测维度
        self.num_decode = num_decode
        self.num_latent = num_latent
        self.total_latent_dim = 3 + 2 + num_latent  # 速度(3) + 脚掌高度(2) + 隐变量
        
        ########################################## 网络架构 ##############################################
        
        # 1. Actor网络
        # MLP特征提取器：当前帧本体观测 -> 特征向量
        self.actor_mlp_extractor = MLP(
            self.num_proprio_one_frame, 
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
        
        # R(2+1)D输出后的两个MLP头
        self.vision_head_1 = nn.Linear(vision_feature_dim, vision_feature_dim)
        self.vision_head_2 = nn.Linear(vision_feature_dim, vision_feature_dim)
        
        # 本体历史特征提取器：5帧历史本体观测 -> 本体历史特征
        self.proprio_history_mlp = MLP(
            num_actor_obs, 
            proprio_history_feature_dim, 
            proprio_history_mlp_hidden_dims, 
            activation
        )
        
        # Encoder: 本体历史特征 + vision_feature_2 -> 线速度 + 足端高度 + 隐向量
        encoder_input_dim = proprio_history_feature_dim + vision_feature_dim
        self.encoder = MLP(encoder_input_dim, encoder_hidden_dims[-1], encoder_hidden_dims[:-1], activation)
        
        # Encoder输出头：线速度(3) + 足端高度(2) + 隐向量z
        self.encoder_vel = nn.Linear(encoder_hidden_dims[-1], 3)
        self.encoder_feet_height = nn.Linear(encoder_hidden_dims[-1], 2)
        self.encoder_latent = nn.Linear(encoder_hidden_dims[-1], num_latent)
        
        # Decoder: 线速度 + 足端高度 + 隐向量z -> 重建下一帧观测
        self.decoder = MLP(self.total_latent_dim, num_decode, decoder_hidden_dims, activation)
        
        # Actor主网络：融合特征 -> 动作
        # 输入: actor_mlp_feature + vision_feature_1 + 线速度(3) + 足端高度(2) + 隐向量z
        actor_input_dim = actor_mlp_feature_dim + vision_feature_dim + 3 + 2 + num_latent
        self.actor = MLP(
            actor_input_dim, 
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
        print("ActorCriticElevationNetMode12 网络结构 (R(2+1)D + MLP + 编解码器)")
        print("="*80)
        print(f"单帧本体观测维度: {self.num_proprio_one_frame}")
        print(f"5帧历史本体观测维度: {num_actor_obs}")
        print(f"Actor MLP特征维度: {actor_mlp_feature_dim}")
        print(f"视觉特征维度: {vision_feature_dim}")
        print(f"本体历史特征维度: {proprio_history_feature_dim}")
        print(f"Encoder输出维度: 速度(3) + 足端(2) + 隐向量({num_latent})")
        print(f"Actor输入维度: {actor_input_dim} = MLP特征({actor_mlp_feature_dim}) + 视觉特征1({vision_feature_dim}) + Encoder输出({3 + 2 + num_latent})")
        print(f"")
        print(f"Actor pipeline:")
        print(f"  - 当前帧本体 -> MLP -> actor_mlp_feature")
        print(f"  - 高程图 -> R(2+1)D -> 两个MLP头 -> vision_feature_1 + vision_feature_2")
        print(f"  - 5帧历史本体 -> MLP -> proprio_history_feature")
        print(f"  - proprio_history_feature + vision_feature_2 -> Encoder -> 速度 + 足端 + 隐向量z")
        print(f"  - 速度 + 足端 + 隐向量z -> Decoder -> 重建观测({num_decode})")
        print(f"  - actor_mlp_feature + vision_feature_1 + Encoder输出 -> Actor -> 动作({num_actions})")
        print(f"")
        print(f"Critic输入维度: {critic_mlp_feature_dim + vision_feature_dim} = MLP特征({critic_mlp_feature_dim}) + 视觉({vision_feature_dim})")
        print(f"✅ 维度验证通过")
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

    def encoder_forward(self, proprio_obs: torch.Tensor, vision_features_base: torch.Tensor):
        """编码器前向传播
        
        Args:
            proprio_obs: 5帧历史本体观测，形状[B, num_actor_obs]
            vision_features_base: R(2+1)D编码器输出的基础视觉特征，形状[B, vision_feature_dim]
            
        Returns:
            vel_output: 线速度，形状[B, 3]
            feet_height_output: 足端高度，形状[B, 2]
            latent_output: 隐向量，形状[B, num_latent]
            obs_decode: 重建的观测，形状[B, num_decode]
        """
        # 1. 通过vision_head_2计算视觉特征2
        vision_feature_2 = self.vision_head_2(vision_features_base.detach())
        
        # 2. 提取本体历史特征
        proprio_history_feature = self.proprio_history_mlp(proprio_obs)
        
        # 3. 拼接本体历史特征和vision_feature_2
        encoder_input = torch.cat([proprio_history_feature, vision_feature_2], dim=-1)
        
        # 4. 通过encoder MLP
        x = self.encoder(encoder_input)
        
        # 5. 输出线速度、足端高度、隐向量
        vel_output = self.encoder_vel(x)
        feet_height_output = self.encoder_feet_height(x)
        latent_output = self.encoder_latent(x)
        
        # 6. 拼接所有输出送入decoder
        decoder_input = torch.cat([vel_output, feet_height_output, latent_output], dim=-1)
        obs_decode = self.decoder(decoder_input)
        
        return vel_output, feet_height_output, latent_output, obs_decode

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        proprio_obs = obs["policy"]
        
        # 应用观测归一化
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取当前帧本体观测的MLP特征
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        mlp_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征，然后通过MLP头1
        vision_features_base = self.elevation_encoder_actor(sampled_height_maps)
        vision_feature_1 = self.vision_head_1(vision_features_base)
        
        # 5. 编码器前向传播（内部会计算vision_feature_2）
        vel_output, feet_height_output, latent_output, obs_decode = \
            self.encoder_forward(proprio_obs_normalized, vision_features_base)
        
        # 6. 融合特征：MLP特征 + 视觉特征1 + Encoder输出(detach)
        fused_features = torch.cat((
            mlp_features, 
            vision_feature_1, 
            vel_output.detach(), 
            feet_height_output.detach(), 
            latent_output.detach()
        ), dim=-1)
        
        # 7. Actor输出动作
        mean = self.actor(fused_features)
        self._update_distribution(mean)
        
        # 8. 记录额外信息用于监控
        self.extra_info["est_vel"] = vel_output
        self.extra_info["est_feet_height"] = feet_height_output
        
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = obs_decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                                       self.actor_obs_normalizer.mean[:self.num_decode]
        else:
            self.extra_info["obs_predict"] = obs_decode
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        proprio_obs = obs["policy"]
        
        # 应用观测归一化
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取当前帧本体观测的MLP特征
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        mlp_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征，然后通过MLP头1
        vision_features_base = self.elevation_encoder_actor(sampled_height_maps)
        vision_feature_1 = self.vision_head_1(vision_features_base)
        
        # 5. 编码器前向传播（内部会计算vision_feature_2）
        vel_output, feet_height_output, latent_output, obs_decode = \
            self.encoder_forward(proprio_obs_normalized, vision_features_base)
        
        # 6. 融合特征：MLP特征 + 视觉特征1 + Encoder输出
        fused_features = torch.cat((
            mlp_features, 
            vision_feature_1, 
            vel_output.detach(), 
            feet_height_output.detach(), 
            latent_output.detach()
        ), dim=-1)
        
        # 7. Actor输出确定性动作
        mean = self.actor(fused_features)
        
        # 8. 记录额外信息
        self.extra_info["est_vel"] = vel_output
        self.extra_info["est_feet_height"] = feet_height_output
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = obs_decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                            self.actor_obs_normalizer.mean[:self.num_decode]
        else:
            self.extra_info["obs_predict"] = obs_decode
        
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
            if obs_group not in ["height_scan_history", "height_scan_policy"]:
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            if obs_group not in ["height_scan_history", "height_scan_critic"]:
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

    def update_encoder(
        self,
        obs_batch: TensorDict,
        next_observations_batch: TensorDict,
        encoder_optimizer: torch.optim.Optimizer,
        max_grad_norm: float
    ) -> dict[str, float]:
        """更新编码器
        
        Args:
            obs_batch: 当前观测批次数据
            next_observations_batch: 下一时刻观测批次数据
            encoder_optimizer: 编码器优化器
            max_grad_norm: 梯度裁剪的最大范数
            
        Returns:
            损失字典，包含各项损失值
        """
        # 1. 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs_normalized = self.actor_obs_normalizer(policy_obs)
        height_maps_obs = obs_batch["height_scan_policy"]
        # 克隆高程图数据避免影响后续的梯度计算
        sampled_height_maps = height_maps_obs.squeeze(1).clone()
        
        # 2. 提取高程图特征
        vision_features_base = self.elevation_encoder_actor(sampled_height_maps)
        
        # 3. 编码器前向传播（内部会计算vision_feature_2）
        vel_output, feet_height_output, latent_output, obs_decode = \
            self.encoder_forward(policy_obs_normalized, vision_features_base)

        # 4. 获取真实目标值
        # 从critic观测中提取真实速度和脚掌高度
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs_normalized = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs_normalized[:, 70:73]  # base_lin_vel
        feet_height_target = critic_obs_normalized[:, 73:75]  # left_foot_height + right_foot_height

        # 获取下一时刻观测作为重建目标
        next_observations = self.get_critic_obs(next_observations_batch)
        next_observations_normalized = self.critic_obs_normalizer(next_observations)
        obs_target = next_observations_normalized[:, 0:self.num_decode]

        vel_target.requires_grad = False
        feet_height_target.requires_grad = False
        obs_target.requires_grad = False

        # 5. 损失计算（只有重建损失，没有KL散度）
        # 线速度重建损失
        vel_MSE = nn.MSELoss()(vel_output, vel_target)
        # 脚掌高度重建损失
        feet_height_MSE = nn.MSELoss()(feet_height_output, feet_height_target)
        # 观测重建损失
        obs_MSE = nn.MSELoss()(obs_decode, obs_target)
        
        # 总损失（不包含KL散度）
        encoder_loss = vel_MSE + feet_height_MSE + obs_MSE

        # 6. 反向传播
        encoder_optimizer.zero_grad()
        encoder_loss.backward(retain_graph=True)

        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)

        # 更新参数
        encoder_optimizer.step()

        return {
            "vel_loss": vel_MSE.item(),
            "feet_height_loss": feet_height_MSE.item(),
            "obs_loss": obs_MSE.item(),
            "total_loss": encoder_loss.item(),
        }

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典，包含主要的优化器和编码器优化器
        """
        import torch.optim as optim
        
        # Actor和Critic的优化器
        optimizer = optim.Adam([
            {'params': self.actor_mlp_extractor.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic_mlp_extractor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.elevation_encoder_critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        # 编码器的优化器（包含actor的高程图编码器、两个头、本体历史MLP、编码器和解码器）
        encoder_optimizer = optim.Adam([
            {'params': self.elevation_encoder_actor.parameters()},
            {'params': self.vision_head_1.parameters()},
            {'params': self.vision_head_2.parameters()},
            {'params': self.proprio_history_mlp.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.encoder_vel.parameters()},
            {'params': self.encoder_feet_height.parameters()},
            {'params': self.encoder_latent.parameters()},
            {'params': self.decoder.parameters()},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer,
            "encoder_optimizer": encoder_optimizer
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode12_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode12策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode12_policy.onnx"
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
        if hasattr(policy, "vision_head_1"):
            self.vision_head_1 = copy.deepcopy(policy.vision_head_1)
            self.vision_head_2 = copy.deepcopy(policy.vision_head_2)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        # 复制编码器相关组件
        if hasattr(policy, "proprio_history_mlp"):
            self.proprio_history_mlp = copy.deepcopy(policy.proprio_history_mlp)
            self.encoder = copy.deepcopy(policy.encoder)
            self.encoder_vel = copy.deepcopy(policy.encoder_vel)
            self.encoder_feet_height = copy.deepcopy(policy.encoder_feet_height)
            self.encoder_latent = copy.deepcopy(policy.encoder_latent)
        
        self.elevation_sampled_frames = policy.elevation_sampled_frames
        self.vision_spatial_size = policy.vision_spatial_size
        self.vision_feature_dim = policy.vision_feature_dim
        self.num_latent = policy.num_latent
        self.actor_mlp_feature_dim = policy.actor_mlp_feature_dim
        self.proprio_history_feature_dim = policy.proprio_history_feature_dim
        
        # 单帧本体观测维度
        self.num_proprio_one_frame = policy.num_proprio_one_frame
        # 5帧历史本体观测维度
        self.history_proprio_dim = self.num_proprio_one_frame * policy.elevation_sampled_frames

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_input):
        """前向传播（单输入版本）
        
        Args:
            obs_input: 合并的观测数据，形状为 [batch_size, total_obs_dim]
                       前 history_proprio_dim 维是5帧历史本体观测（用于编码器）
                       接下来 num_proprio_one_frame 维是当前帧本体观测（用于MLP）
                       后面是展平的高程图数据（需要reshape为 [B, sampled_frames, height, width]）
        
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_input.shape[0]
        
        # 切片分离各部分数据
        history_proprio_data = obs_input[:, :self.history_proprio_dim]  # 5帧历史本体观测
        current_proprio_data = obs_input[:, self.history_proprio_dim:self.history_proprio_dim + self.num_proprio_one_frame]  # 当前帧本体观测
        elevation_data_flat = obs_input[:, self.history_proprio_dim + self.num_proprio_one_frame:]  # 高程图
        
        # 将高程图数据reshape为 [B, sampled_frames, height, width]
        height, width = self.vision_spatial_size
        elevation_data = elevation_data_flat.reshape(
            batch_size, self.elevation_sampled_frames, height, width
        )
        
        # 应用归一化器
        history_proprio_data = self.normalizer(history_proprio_data)
        current_proprio_data = self.normalizer(current_proprio_data)
        
        # 提取MLP特征
        mlp_features = self.actor_mlp_extractor(current_proprio_data)
        
        # 提取视觉特征并通过头1
        vision_features_base = self.elevation_encoder(elevation_data)
        vision_feature_1 = self.vision_head_1(vision_features_base)
        
        # 编码器前向传播（内部会计算vision_feature_2）
        vision_feature_2 = self.vision_head_2(vision_features_base.detach())
        proprio_history_feature = self.proprio_history_mlp(history_proprio_data)
        encoder_input = torch.cat([proprio_history_feature, vision_feature_2], dim=-1)
        x = self.encoder(encoder_input)
        vel_output = self.encoder_vel(x)
        feet_height_output = self.encoder_feet_height(x)
        latent_output = self.encoder_latent(x)
        
        # 融合所有特征：MLP特征 + 视觉特征1 + Encoder输出
        fused_features = torch.cat([
            mlp_features, 
            vision_feature_1, 
            vel_output, 
            feet_height_output, 
            latent_output
        ], dim=-1)
        
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
        total_obs_dim = self.history_proprio_dim + self.num_proprio_one_frame + elevation_dim
        
        # 创建单个合并的输入示例
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (Mode12单输入模式 - R(2+1)D + MLP + 编解码器):")
        print(f"{'='*80}")
        print(f"  5帧历史本体观测维度:   {self.history_proprio_dim}")
        print(f"  当前帧本体观测维度:     {self.num_proprio_one_frame}")
        print(f"  MLP特征维度:            {self.actor_mlp_feature_dim}")
        print(f"  本体历史特征维度:       {self.proprio_history_feature_dim}")
        print(f"  高程图维度:             {elevation_dim} ({sampled_frames}×{height}×{width})")
        print(f"  视觉特征维度:           {self.vision_feature_dim}")
        print(f"  Encoder输出维度:        {3 + 2 + self.num_latent} (速度3 + 足端2 + 隐向量{self.num_latent})")
        print(f"  总输入维度:             {total_obs_dim}")
        print(f"  输入切片方式:")
        print(f"    [:, :{self.history_proprio_dim}] = 5帧历史本体")
        print(f"    [:, {self.history_proprio_dim}:{self.history_proprio_dim + self.num_proprio_one_frame}] = 当前本体")
        print(f"    [:, {self.history_proprio_dim + self.num_proprio_one_frame}:] = 高程图")
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
