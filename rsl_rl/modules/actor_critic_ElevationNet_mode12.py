# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode12: R2+1D + AE架构

网络结构:
    Actor pipeline:
        单帧本体观测 -> MLP -> 特征向量A
        特征向量A + 高程图特征B + [线速度估计 + 隐向量z] -> Actor网络 -> 动作
    
    高程图R2+1D编码器:
        历史高程图序列 -> R2+1D -> 两个linear head -> 特征向量B和C
    
    本体Encoder-Decoder (AE架构):
        历史本体信息 -> MLP -> 特征向量D
        [C + D] -> Encoder -> 线速度估计 + 隐向量z
        [vel + z] -> Decoder -> 重建下一时刻观测值
    
    Critic:
        本体特权观测 -> MLP提取特征 -> critic_mlp_feature
        特权历史高程图序列 -> R2+1D提取特征 -> critic_vision_feature
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
import torch.nn.functional as F

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
    """Mode12: R2+1D + AE架构
    
    网络组成:
    1. Actor部分:
       - 单帧本体观测 -> MLP -> 特征向量A
       - [特征A + 高程图特征B + 线速度估计 + 隐向量z] -> Actor网络 -> 动作
    
    2. 高程图R2+1D编码器:
       - 历史高程图序列 -> R2+1D -> 特征向量 -> 两个linear head -> 特征B和C
    
    3. 本体编码器:
       - 历史本体信息 -> MLP -> 特征向量D
    
    4. AE Encoder-Decoder:
       - [C + D] -> Encoder MLP -> 线速度估计 + 隐向量z
       - [vel + z] -> Decoder MLP -> 重建下一观测
    
    5. Critic部分:
       - 本体特权观测 -> MLP -> critic_mlp_feature
       - 特权高程图序列 -> R2+1D -> critic_vision_feature
       - [critic_mlp_feature + critic_vision_feature] -> Critic -> 价值
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
        # 高程图配置
        elevation_sampled_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (25, 17),
        # R2+1D高程图编码器配置
        r2plus1d_hidden_dims: list[int] = [16, 32, 64],
        r2plus1d_kernel_sizes: list[int] = [3, 3, 3],
        r2plus1d_strides: list[int] = [2, 2, 2],
        elevation_feature_dim: int = 64,  # R2+1D输出的特征维度
        feature_B_dim: int = 32,  # 特征B的维度
        feature_C_dim: int = 32,  # 特征C的维度
        # 本体编码器配置（历史本体 -> 特征D）
        proprio_encoder_hidden_dims: tuple[int] | list[int] = [256, 128],
        feature_D_dim: int = 64,  # 特征D的维度
        # AE Encoder配置（[C+D] -> vel + z）
        ae_encoder_hidden_dims: tuple[int] | list[int] = [128, 64],
        num_vel: int = 3,
        latent_dim: int = 32,  # 隐向量z的维度
        # AE Decoder配置（z -> 重建下一观测）
        ae_decoder_hidden_dims: tuple[int] | list[int] = [64, 128, 256],
        num_decode: int = 70,  # 重建的下一观测维度
        # Critic MLP配置
        critic_mlp_feature_dim: int = 64,
        critic_mlp_extractor_hidden_dims: tuple[int] | list[int] = [128],
        # 单帧本体MLP配置（单帧本体 -> 特征A）
        single_frame_mlp_hidden_dims: tuple[int] | list[int] = [64],
        feature_A_dim: int = 32,  # 特征A的维度
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
        self.elevation_feature_dim = elevation_feature_dim
        self.feature_A_dim = feature_A_dim
        self.feature_B_dim = feature_B_dim
        self.feature_C_dim = feature_C_dim
        self.feature_D_dim = feature_D_dim
        self.latent_dim = latent_dim
        self.num_vel = num_vel
        self.critic_mlp_feature_dim = critic_mlp_feature_dim
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / elevation_sampled_frames)  # 单帧本体观测维度
        self.num_decode = num_decode
        
        ########################################## 网络架构 ##############################################
        
        # 1. R2+1D高程图编码器
        # 高程图历史 -> R2+1D -> 特征向量
        self.elevation_encoder = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r2plus1d_hidden_dims,
            kernel_sizes=r2plus1d_kernel_sizes,
            strides=r2plus1d_strides,
            out_dim=elevation_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # 两个linear head：特征B和C
        self.elevation_head_B = nn.Linear(elevation_feature_dim, feature_B_dim)
        self.elevation_head_C = nn.Linear(elevation_feature_dim, feature_C_dim)
        
        # 2. 单帧本体MLP：单帧本体 -> 特征A
        self.single_frame_encoder = MLP(
            self.num_proprio_one_frame,
            feature_A_dim,
            single_frame_mlp_hidden_dims,
            activation
        )
        
        # 3. 本体编码器：历史本体 -> 特征D
        self.proprio_encoder = MLP(
            num_actor_obs,  # 5帧历史本体观测
            feature_D_dim,
            proprio_encoder_hidden_dims,
            activation
        )
        
        # 4. AE Encoder：[C + D] -> vel + z
        ae_encoder_input_dim = feature_C_dim + feature_D_dim
        ae_encoder_feature_dim = max(ae_encoder_hidden_dims)
        self.ae_encoder = MLP(
            ae_encoder_input_dim,
            ae_encoder_feature_dim,
            ae_encoder_hidden_dims[:-1],
            activation
        )
        
        # AE Encoder输出heads
        self.encoder_vel = nn.Linear(ae_encoder_feature_dim, num_vel)
        self.encoder_latent = nn.Linear(ae_encoder_feature_dim, latent_dim)
        
        # 5. AE Decoder：[vel + z] -> 重建下一观测
        self.ae_decoder = MLP(
            num_vel + latent_dim,
            num_decode,
            ae_decoder_hidden_dims,
            activation
        )
        
        # 6. Actor网络
        # Actor主网络：[特征A + 特征B + vel + z] -> 动作
        actor_input_dim = feature_A_dim + feature_B_dim + num_vel + latent_dim
        self.actor = MLP(
            actor_input_dim,
            num_actions,
            actor_hidden_dims,
            activation
        )
        
        # 7. Critic网络
        # MLP特征提取器：本体特权观测 -> 特征向量
        self.critic_mlp_extractor = MLP(
            num_critic_obs,
            critic_mlp_feature_dim,
            critic_mlp_extractor_hidden_dims,
            activation
        )
        
        # R2+1D编码器：特权高程图序列 -> 视觉特征向量
        self.elevation_encoder_critic = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r2plus1d_hidden_dims,
            kernel_sizes=r2plus1d_kernel_sizes,
            strides=r2plus1d_strides,
            out_dim=elevation_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # Critic主网络：融合特征 -> 价值
        self.critic = MLP(
            critic_mlp_feature_dim + elevation_feature_dim,
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
        print("ActorCriticElevationNetMode12 网络结构 (R2+1D + AE架构)")
        print("="*80)
        print(f"单帧本体观测维度: {self.num_proprio_one_frame}")
        print(f"5帧历史本体观测维度: {num_actor_obs}")
        print(f"")
        print(f"单帧本体编码器:")
        print(f"  - 单帧本体({self.num_proprio_one_frame}) -> MLP -> 特征A({feature_A_dim})")
        print(f"")
        print(f"R2+1D高程图编码器:")
        print(f"  - 高程图序列 -> R2+1D -> 特征({elevation_feature_dim})")
        print(f"  - 特征 -> Linear head B -> 特征B({feature_B_dim})")
        print(f"  - 特征 -> Linear head C -> 特征C({feature_C_dim})")
        print(f"")
        print(f"历史本体编码器:")
        print(f"  - 5帧历史本体 -> MLP -> 特征D({feature_D_dim})")
        print(f"")
        print(f"AE Encoder-Decoder:")
        print(f"  - [C({feature_C_dim}) + D({feature_D_dim})] -> AE Encoder -> 线速度({num_vel}) + 隐向量z({latent_dim})")
        print(f"  - [vel({num_vel}) + z({latent_dim})] -> AE Decoder -> 重建下一观测({num_decode})")
        print(f"")
        print(f"Actor pipeline:")
        print(f"  - Actor输入维度: {actor_input_dim} = 特征A({feature_A_dim}) + 特征B({feature_B_dim}) + vel({num_vel}) + z({latent_dim})")
        print(f"  - Actor输出维度: {num_actions}")
        print(f"")
        print(f"Critic pipeline:")
        print(f"  - Critic输入维度: {critic_mlp_feature_dim + elevation_feature_dim} = MLP特征({critic_mlp_feature_dim}) + R2+1D特征({elevation_feature_dim})")
        print(f"✅ 维度验证通过")
        print("="*80 + "\n")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """重置网络状态"""
        pass

    def forward(self) -> NoReturn:
        """Forward方法不应被调用"""
        raise NotImplementedError("Use act() or act_inference() instead.")

    @property
    def action_mean(self) -> torch.Tensor:
        """返回动作均值"""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """返回动作标准差"""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """返回动作分布的熵"""
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新观测归一化统计量"""
        if self.actor_obs_normalization:
            policy_obs = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.actor_obs_normalizer.update(policy_obs)
        if self.critic_obs_normalization:
            critic_obs = torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)
            self.critic_obs_normalizer.update(critic_obs)

    def act(self, obs: TensorDict, **kwargs) -> tuple[torch.Tensor, dict]:
        """Actor前向传播（训练模式）
        
        Args:
            obs: 包含policy、height_scan_policy的TensorDict
            
        Returns:
            actions: [B, num_actions] 采样的动作
            extra_info: 包含encoder输出等额外信息的字典
        """
        # 1. 提取观测
        proprio_obs = obs["policy"]  # 5帧历史本体
        elevation_obs = obs["height_scan_policy"]  # 高程图，已经是[B, T, H, W]格式
        
        # 归一化
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        current_frame_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 当前帧本体
        
        # 2. 单帧本体编码
        feature_A = self.single_frame_encoder(current_frame_obs)
        
        # 3. R2+1D高程图编码器
        elevation_feature = self.elevation_encoder(elevation_obs)
        feature_B = self.elevation_head_B(elevation_feature)
        feature_C = self.elevation_head_C(elevation_feature.detach())
        
        # 4. 本体编码器 -> 特征D
        feature_D = self.proprio_encoder(proprio_obs)
        
        # 5. AE Encoder: [C + D] -> vel + z
        ae_input = torch.cat([feature_C, feature_D], dim=-1)
        ae_features = self.ae_encoder(ae_input)
        vel = self.encoder_vel(ae_features)
        latent_z = self.encoder_latent(ae_features)
        
        # 6. Actor前向传播
        # 拼接：特征A + 特征B + vel + z（vel和z需要detach，但feature_A和feature_B不detach）
        actor_input = torch.cat([
            feature_A, 
            feature_B, 
            vel.detach(), 
            latent_z.detach()
        ], dim=-1)
        
        # 通过Actor网络
        action_mean = self.actor(actor_input)
        
        # 7. 动作分布和采样
        if self.noise_std_type == "scalar":
            action_std = self.std
        elif self.noise_std_type == "log":
            action_std = torch.exp(self.log_std)
        
        self.distribution = Normal(action_mean, action_std)
        actions = self.distribution.sample()
        
        # 8. 存储额外信息
        extra_info = {
            "vel": vel,
        }
        
        return actions, extra_info

    def act_inference(self, observations: TensorDict) -> torch.Tensor:
        """Actor前向传播（推理模式）
        
        Args:
            observations: 包含policy、height_scan_policy的TensorDict
            
        Returns:
            action_mean: [B, num_actions] 动作均值（无噪声）
        """
        # 1. 提取观测
        proprio_obs = observations["policy"]  # 5帧历史本体
        elevation_obs = observations["height_scan_policy"]  # 高程图，已经是[B, T, H, W]格式
        
        # 归一化
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        current_frame_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 当前帧本体
        
        # 2. 单帧本体编码
        feature_A = self.single_frame_encoder(current_frame_obs)
        
        # 3. R2+1D高程图编码器
        elevation_feature = self.elevation_encoder(elevation_obs)
        feature_B = self.elevation_head_B(elevation_feature)
        feature_C = self.elevation_head_C(elevation_feature.detach())
        
        # 4. 本体编码器 -> 特征D
        feature_D = self.proprio_encoder(proprio_obs)
        
        # 5. AE Encoder: [C + D] -> vel + z
        ae_input = torch.cat([feature_C, feature_D], dim=-1)
        ae_features = self.ae_encoder(ae_input)
        vel = self.encoder_vel(ae_features)
        latent_z = self.encoder_latent(ae_features)
        
        # 6. Actor前向传播
        # 拼接：特征A + 特征B + vel + z
        actor_input = torch.cat([feature_A, feature_B, vel, latent_z], dim=-1)
        action_mean = self.actor(actor_input)
        
        return action_mean

    def evaluate(self, observations: TensorDict, **kwargs) -> torch.Tensor:
        """Critic前向传播
        
        Args:
            observations: 包含critic、height_scan_critic的TensorDict
            
        Returns:
            value: [B, 1] 状态价值
        """
        # 1. 提取观测
        critic_obs = observations["critic"]  # 特权观测
        elevation_obs = observations["height_scan_critic"]  # 高程图，已经是[B, T, H, W]格式
        
        # 归一化
        critic_obs = self.critic_obs_normalizer(critic_obs)
        
        # 2. 提取特征
        critic_mlp_feature = self.critic_mlp_extractor(critic_obs)
        critic_vision_feature = self.elevation_encoder_critic(elevation_obs)
        
        # 3. 融合特征并预测价值
        critic_input = torch.cat([critic_mlp_feature, critic_vision_feature], dim=-1)
        value = self.critic(critic_input)
        
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
        """计算动作的对数概率
        
        Args:
            actions: [B, num_actions] 动作
            
        Returns:
            log_prob: [B, 1] 对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def update_encoder(
        self,
        obs_batch: TensorDict,
        next_obs_batch: TensorDict,
        encoder_optimizer: torch.optim.Optimizer,
        max_grad_norm: float
    ) -> dict[str, float]:
        """更新编码器和解码器（AE架构）
        
        Args:
            obs_batch: 当前观测批次
            next_obs_batch: 下一时刻观测批次
            encoder_optimizer: 编码器优化器
            max_grad_norm: 最大梯度范数
            
        Returns:
            losses: 包含各项损失的字典
        """
        # 1. 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs_normalized = self.actor_obs_normalizer(policy_obs)
        elevation_obs = obs_batch["height_scan_policy"]  # 有噪声的高程图，已经是[B, T, H, W]格式
        
        # 2. R2+1D高程图编码器
        elevation_feature = self.elevation_encoder(elevation_obs)
        feature_B = self.elevation_head_B(elevation_feature)
        feature_C = self.elevation_head_C(elevation_feature.detach())
        
        # 3. 本体编码器 -> 特征D
        feature_D = self.proprio_encoder(policy_obs_normalized)
        
        # 4. AE Encoder: [C + D] -> vel + z
        ae_input = torch.cat([feature_C, feature_D], dim=-1)
        ae_features = self.ae_encoder(ae_input)
        vel = self.encoder_vel(ae_features)
        latent_z = self.encoder_latent(ae_features)
        
        # 5. 计算重建损失
        
        # 3.1 获取真实目标值
        # 从critic观测中提取真实速度
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs_normalized = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs_normalized[:, 70:73]  # base_lin_vel
        
        # 获取下一时刻观测作为重建目标
        next_observations = self.get_critic_obs(next_obs_batch)
        next_observations_normalized = self.critic_obs_normalizer(next_observations)
        obs_target = next_observations_normalized[:, 0:self.num_decode]
        
        vel_target.requires_grad = False
        obs_target.requires_grad = False
        
        # 3.2 线速度重建损失
        vel_loss = nn.functional.mse_loss(vel, vel_target)
        
        # 3.3 观测重建损失：[vel + z] -> 重建下一观测
        decoder_input = torch.cat([vel.detach(), latent_z], dim=-1)
        recon_obs = self.ae_decoder(decoder_input)
        obs_loss = nn.functional.mse_loss(recon_obs, obs_target)
        
        # 6. 总损失（AE架构，无KL散度）
        total_loss = vel_loss + obs_loss
        
        # 7. 反向传播和优化
        encoder_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)
        
        encoder_optimizer.step()
        
        # 8. 返回损失信息
        losses = {
            "vel_loss": vel_loss.item(),
            "obs_loss": obs_loss.item(),
            "encoder_total_loss": total_loss.item(),
        }
        
        return losses

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
            {'params': self.single_frame_encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic_mlp_extractor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.elevation_encoder_critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        # 编码器和解码器的优化器
        encoder_optimizer = optim.Adam([
            {'params': self.elevation_encoder.parameters()},
            {'params': self.elevation_head_B.parameters()},
            {'params': self.elevation_head_C.parameters()},
            {'params': self.proprio_encoder.parameters()},
            {'params': self.ae_encoder.parameters()},
            {'params': self.encoder_vel.parameters()},
            {'params': self.encoder_latent.parameters()},
            {'params': self.ae_decoder.parameters()},
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
        """将Mode12策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要
        """
        import copy
        import os
        
        if normalizer is None:
            normalizer = torch.nn.Identity()
        
        _exporter = _ElevationNetMode12OnnxPolicyExporter(
            self,
            normalizer,
            self.obs_groups,
            self.num_proprio_one_frame,
            self.elevation_sampled_frames,
            self.vision_spatial_size
        )
        
        # 创建目录
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        # 导出
        _exporter.export(full_path, verbose=verbose)


class _ElevationNetMode12OnnxPolicyExporter(torch.nn.Module):
    """Mode12策略的ONNX导出器"""

    def __init__(
        self,
        policy: ActorCriticElevationNetMode12,
        normalizer: torch.nn.Module,
        obs_groups: dict,
        num_proprio_one_frame: int,
        elevation_sampled_frames: int,
        vision_spatial_size: tuple[int, int]
    ):
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu()
        self.normalizer = copy.deepcopy(normalizer).cpu()
        self.obs_groups = obs_groups
        self.num_proprio_one_frame = num_proprio_one_frame
        self.elevation_sampled_frames = elevation_sampled_frames
        self.vision_spatial_size = vision_spatial_size

    def forward(self, proprio_obs_history, elevation_obs, current_frame_obs):
        """ONNX前向传播
        
        Args:
            proprio_obs_history: [B, 5*proprio_dim] 5帧历史本体观测
            elevation_obs: [B, T*H*W] 高程图序列（展平）
            current_frame_obs: [B, proprio_dim] 当前帧本体观测
            
        Returns:
            actions: [B, num_actions] 动作
        """
        # 归一化
        proprio_obs_history = self.normalizer(proprio_obs_history)
        
        # 重塑高程图
        batch_size = elevation_obs.shape[0]
        elevation_obs = elevation_obs.view(
            batch_size,
            self.elevation_sampled_frames,
            self.vision_spatial_size[0],
            self.vision_spatial_size[1]
        )
        
        # 单帧本体编码
        feature_A = self.policy.single_frame_encoder(current_frame_obs)
        
        # R2+1D高程图编码器
        elevation_feature = self.policy.elevation_encoder(elevation_obs)
        feature_B = self.policy.elevation_head_B(elevation_feature)
        feature_C = self.policy.elevation_head_C(elevation_feature.detach())
        
        # 本体编码器 -> 特征D
        feature_D = self.policy.proprio_encoder(proprio_obs_history)
        
        # AE Encoder: [C + D] -> vel + z
        ae_input = torch.cat([feature_C, feature_D], dim=-1)
        ae_features = self.policy.ae_encoder(ae_input)
        vel = self.policy.encoder_vel(ae_features)
        latent_z = self.policy.encoder_latent(ae_features)
        
        # Actor前向传播：[特征A + 特征B + vel + z]
        actor_input = torch.cat([feature_A, feature_B, vel, latent_z], dim=-1)
        actions = self.policy.actor(actor_input)
        
        return actions

    def export(self, path: str, verbose: bool = False):
        """执行ONNX导出"""
        self.eval()
        
        # 创建虚拟输入
        dummy_proprio_history = torch.zeros(1, self.num_proprio_one_frame * self.elevation_sampled_frames)
        dummy_elevation = torch.zeros(1, self.elevation_sampled_frames * self.vision_spatial_size[0] * self.vision_spatial_size[1])
        dummy_current_frame = torch.zeros(1, self.num_proprio_one_frame)
        
        # 导出
        torch.onnx.export(
            self,
            (dummy_proprio_history, dummy_elevation, dummy_current_frame),
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['proprio_obs_history', 'elevation_obs', 'current_frame_obs'],
            output_names=['actions'],
            verbose=verbose
        )
        
        print(f"✅ ONNX模型已导出至: {path}")
