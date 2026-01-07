# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode11: R(2+1)D CNN + VAE架构

网络结构:
    Actor pipeline:
        当前帧本体观测(102维) + R(2+1)D提取的高程图特征(64维) + 
        线速度(3维) + 足端高度(2维) + 隐向量z(可配置维度) -> Actor MLP -> 动作
    
    VAE pipeline:
        5帧历史本体观测(510维) + detach后的高程图特征(64维) -> VAE encoder -> 
        线速度(3维) + 左右足端高度(2维) + 隐向量z(可配置维度)
        
        线速度 + 足端高度 + 隐向量z -> VAE decoder -> 重建下一帧观测
    
    Critic:
        高程图序列 -> R(2+1)D CNN -> 特征 + 本体观测 -> Critic MLP -> 价值

R(2+1)D CNN说明:
    将3D卷积分解为2D空间卷积 + 1D时间卷积
    每个block: 2D Conv(空间) -> 1D Conv(时间) -> ReLU
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


class R2Plus1DElevationEncoder(nn.Module):
    """R(2+1)D CNN编码器，用于处理采样后的高程图序列
    
    将3D卷积分解为2D空间卷积和1D时间卷积，更高效地捕获时空特征
    每个block包含: 2D Conv(空间) -> 1D Conv(时间) -> ReLU
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
        
        # 构建R(2+1)D卷积块
        blocks = []
        in_channels = 1  # 单通道高程图
        
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            # 2D空间卷积
            spatial_conv = nn.Conv2d(
                in_channels, hidden_dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(kernel_size//2, kernel_size//2)
            )
            
            # 1D时间卷积
            temporal_conv = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )
            
            blocks.append({
                'spatial': spatial_conv,
                'temporal': temporal_conv,
                'relu': nn.ReLU()
            })
            
            in_channels = hidden_dim
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict(block) for block in blocks
        ])
        
        # 计算经过R(2+1)D CNN后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_frames, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self._forward_conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def _forward_conv(self, x):
        """R(2+1)D卷积的前向传播
        
        Args:
            x: [B, T, H, W] 输入序列
            
        Returns:
            x: [B, T', C, H', W'] 输出特征图
        """
        B, T, H, W = x.shape
        
        for i, block in enumerate(self.blocks):
            # 2D空间卷积: 对每一帧独立处理
            if i == 0:
                # 第一层：输入是单通道 [B, T, H, W]
                x_reshaped = x.reshape(B * T, 1, H, W)
            else:
                # 后续层：输入是多通道 [B, T, C, H, W]
                _, T, C, H, W = x.shape
                x_reshaped = x.reshape(B * T, C, H, W)
            
            x_spatial = block['spatial'](x_reshaped)  # [B*T, C_out, H', W']
            
            # 将结果reshape回时间序列
            C_out, H_new, W_new = x_spatial.shape[1:]
            x_spatial = x_spatial.reshape(B, T, C_out, H_new, W_new)
            
            # 1D时间卷积: 对时间维度做卷积
            # 将 [B, T, C, H', W'] 重组为 [B*H'*W', C, T]
            x_temporal_input = x_spatial.permute(0, 3, 4, 2, 1)  # [B, H', W', C, T]
            x_temporal_input = x_temporal_input.reshape(B * H_new * W_new, C_out, T)  # [B*H'*W', C, T]
            x_temporal = block['temporal'](x_temporal_input)  # [B*H'*W', C, T']
            
            # Reshape回 [B, T', C, H', W']
            T_new = x_temporal.shape[-1]
            x_temporal = x_temporal.reshape(B, H_new, W_new, C_out, T_new)  # [B, H', W', C, T']
            x = x_temporal.permute(0, 4, 3, 1, 2)  # [B, T', C, H', W']
            
            # 激活函数
            x = block['relu'](x)
            
            # 更新维度，准备下一层
            T, H, W = T_new, H_new, W_new
        
        return x

    def forward(self, x):
        # x: [B, T, H, W]
        # 归一化 (使用 clamp 而不是 clip 避免原地修改)
        x_mean = x.mean(dim=(-1, -2), keepdim=True)
        x_normalized = (x - x_mean) / 0.1
        x = torch.clamp(x_normalized, -5.0, 5.0)
        
        # R(2+1)D卷积
        x = self._forward_conv(x)
        
        # 展平并全连接
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ActorCriticElevationNetMode11(nn.Module):
    """Mode11: R(2+1)D CNN + VAE架构
    
    网络组成:
    1. 高程图R(2+1)D CNN编码器 - 分解的时空卷积处理高程图序列，输出64维特征
    2. VAE Encoder - 5帧历史本体观测 + detach后的高程图特征 -> 线速度 + 足端高度 + 隐向量z
    3. Actor - 当前帧本体观测 + 高程图特征 + 线速度 + 足端高度 + 隐向量z -> 动作
    4. VAE Decoder - 线速度 + 足端高度 + 隐向量z -> 重建下一帧观测
    5. Critic - 高程图特征 + 本体观测 -> 价值评估
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
        # R(2+1)D CNN配置
        r2p1d_hidden_dims: list[int] = [16, 32, 64],
        r2p1d_kernel_sizes: list[int] = [3, 3, 3],
        # VAE编码器-解码器配置
        encoder_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        decoder_hidden_dims: tuple[int] | list[int] = [128, 256, 512],
        num_latent: int = 14,  # 纯隐向量维度
        num_decode: int = 70,  # decoder重建的观测维度
        VAE_beta: float = 1.0,
        use_beta_annealing: bool = True,
        beta_start: float = 0.01,
        beta_end: float = 0.5,
        beta_anneal_steps: int = 10000,
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
        
        # Beta配置：根据标志位决定使用退火还是固定值
        self.use_beta_annealing = use_beta_annealing
        if self.use_beta_annealing:
            # 从alg_cfg中获取训练配置
            if alg_cfg is not None:
                self.num_learning_epochs = alg_cfg.get('num_learning_epochs', 5)
                self.num_mini_batches = alg_cfg.get('num_mini_batches', 4)
            else:
                self.num_learning_epochs = 5
                self.num_mini_batches = 4
            
            self.updates_per_iteration = self.num_learning_epochs * self.num_mini_batches
            
            # 使用退火策略
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_anneal_steps = beta_anneal_steps
            self.beta = self.beta_start
            self.current_step = 0
            self.update_call_count = 0
            print(f"[Beta Annealing] 已启用 - 起始值: {self.beta_start}, 目标值: {self.beta_end}, 退火步数: {self.beta_anneal_steps}")
            print(f"[Beta Annealing] 训练配置 - epochs: {self.num_learning_epochs}, mini_batches: {self.num_mini_batches}, 每iteration更新次数: {self.updates_per_iteration}")
        else:
            # 使用固定值
            self.beta = VAE_beta
            self.beta_start = VAE_beta
            self.beta_end = VAE_beta
            self.beta_anneal_steps = 0
            self.current_step = 0
            self.update_call_count = 0
            self.num_learning_epochs = 5
            self.num_mini_batches = 4
            self.updates_per_iteration = 20
            print(f"[Beta Fixed] 使用固定beta值: {self.beta}")
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / elevation_sampled_frames)  # 单帧本体观测维度
        self.num_decode = num_decode
        self.num_latent = num_latent
        self.total_latent_dim = 3 + 2 + num_latent  # 速度(3) + 脚掌高度(2) + 纯隐变量
        
        ########################################## 网络架构 ##############################################
        
        # 1. Actor网络 - 使用R(2+1)D CNN
        # Actor输入：当前帧本体观测(102维) + 高程图特征(64维) + 线速度(3) + 足端高度(2) + 隐向量(num_latent)
        actor_input_dim = self.num_proprio_one_frame + vision_feature_dim + 3 + 2 + num_latent
        self.elevation_encoder_actor = R2Plus1DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r2p1d_hidden_dims,
            kernel_sizes=r2p1d_kernel_sizes,
            strides=[2] * len(r2p1d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        
        # 2. Critic网络 - 使用R(2+1)D CNN
        self.elevation_encoder_critic = R2Plus1DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r2p1d_hidden_dims,
            kernel_sizes=r2p1d_kernel_sizes,
            strides=[2] * len(r2p1d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.critic = MLP(num_critic_obs + vision_feature_dim, 1, critic_hidden_dims, activation)
        
        # 3. VAE Encoder
        # VAE输入：5帧历史本体观测 + detach后的高程图特征
        vae_input_dim = num_actor_obs + vision_feature_dim  # 510 + 64 = 574
        self.vae_encoder = MLP(vae_input_dim, encoder_hidden_dims[-1], encoder_hidden_dims[:-1], activation)
        
        # VAE encoder输出：线速度(3) + 足端高度(2) + 隐向量z的均值和方差
        self.encoder_vel = nn.Linear(encoder_hidden_dims[-1], 3)
        self.encoder_feet_height = nn.Linear(encoder_hidden_dims[-1], 2)
        self.encoder_latent_mean = nn.Linear(encoder_hidden_dims[-1], num_latent)
        self.encoder_latent_logvar = nn.Linear(encoder_hidden_dims[-1], num_latent)
        
        # 4. VAE Decoder
        # Decoder输入：线速度(3) + 足端高度(2) + 隐向量z
        self.vae_decoder = MLP(self.total_latent_dim, num_decode, decoder_hidden_dims, activation)

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
        
        # 维度验证
        expected_actor_input = self.num_proprio_one_frame + vision_feature_dim + 3 + 2 + num_latent
        assert actor_input_dim == expected_actor_input, \
            f"Actor input dimension mismatch! Expected {expected_actor_input}, got {actor_input_dim}"
        
        # 打印网络结构
        print("\n" + "="*80)
        print("ActorCriticElevationNetMode11 网络结构 (R(2+1)D CNN + VAE)")
        print("="*80)
        print(f"Actor输入维度: {actor_input_dim} = 本体({self.num_proprio_one_frame}) + 视觉({vision_feature_dim}) + 速度(3) + 足端(2) + 隐向量({num_latent})")
        print(f"Actor pipeline: 当前帧本体观测 + 高程图特征 + VAE输出 -> 动作({num_actions})")
        print(f"VAE encoder: 5帧历史本体观测({num_actor_obs}) + 高程图特征({vision_feature_dim}) -> 线速度(3) + 足端高度(2) + 隐向量z({num_latent})")
        print(f"VAE decoder: 线速度 + 足端高度 + 隐向量z({self.total_latent_dim}) -> 重建观测({num_decode})")
        print(f"✅ 维度验证通过")
        print("="*80)
        print(self)
        print("="*80 + "\n")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def update_beta_schedule(self) -> None:
        """更新beta值的退火调度
        
        单周期退火：从beta_start线性增长到beta_end，然后保持不变
        """
        if not self.use_beta_annealing:
            return
        
        # 增加调用计数
        self.update_call_count += 1
        
        # 只有当调用次数达到一个完整iteration时才更新beta
        if self.update_call_count % self.updates_per_iteration != 0:
            return
        
        # 真正的更新逻辑
        self.current_step += 1
        
        if self.current_step <= self.beta_anneal_steps:
            # 线性退火阶段
            progress = self.current_step / self.beta_anneal_steps
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            # 保持最终值
            self.beta = self.beta_end

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(std)
        code = mean + std * code_temp
        return code

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def vae_forward(self, proprio_obs: torch.Tensor, vision_features: torch.Tensor):
        """VAE前向传播
        
        Args:
            proprio_obs: 5帧历史本体观测，形状[B, 510]
            vision_features: detach后的高程图特征，形状[B, 64]
            
        Returns:
            vel_output: 线速度，形状[B, 3]
            feet_height_output: 足端高度，形状[B, 2]
            latent_sample: 隐向量采样，形状[B, num_latent]
            obs_decode: 重建的观测，形状[B, num_decode]
            latent_mean: 隐向量均值
            latent_logvar: 隐向量对数方差
        """
        # 1. 拼接输入
        vae_input = torch.cat([proprio_obs, vision_features.detach()], dim=-1)
        
        # 2. 通过encoder MLP
        x = self.vae_encoder(vae_input)
        
        # 3. 输出线速度、足端高度、隐向量的均值和方差
        vel_output = self.encoder_vel(x)
        feet_height_output = self.encoder_feet_height(x)
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        
        # 限制方差范围 (使用 clamp 避免原地修改)
        latent_logvar = torch.clamp(latent_logvar, min=-10, max=10)
        
        # 4. 重参数化采样隐向量
        latent_sample = self.reparameterise(latent_mean, latent_logvar)
        
        # 5. 拼接所有输出送入decoder
        decoder_input = torch.cat([vel_output, feet_height_output, latent_sample], dim=-1)
        obs_decode = self.vae_decoder(decoder_input)
        
        return vel_output, feet_height_output, latent_sample, obs_decode, latent_mean, latent_logvar

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        proprio_obs = obs["policy"]
        
        # 应用观测归一化
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1)
        
        # 3. 提取高程图特征 (使用R(2+1)D CNN)
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 4. VAE前向传播（先计算VAE输出）
        vel_output, feet_height_output, latent_sample, obs_decode, latent_mean, latent_logvar = \
            self.vae_forward(proprio_obs_normalized, vision_features)
        
        # 5. Actor输出动作：当前帧本体观测 + 高程图特征 + VAE输出
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        actor_input = torch.cat((vision_features, current_proprio_obs, vel_output.detach(), feet_height_output.detach(), latent_sample.detach()), dim=-1)
        mean = self.actor(actor_input)
        self._update_distribution(mean)
        
        # 6. 记录额外信息用于监控
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
        
        # 3. 提取高程图特征 (使用R(2+1)D CNN)
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 4. VAE前向传播（先计算VAE输出）
        vel_output, feet_height_output, latent_sample, obs_decode, latent_mean, latent_logvar = \
            self.vae_forward(proprio_obs_normalized, vision_features)
        
        # 5. Actor输出确定性动作：当前帧本体观测 + 高程图特征 + VAE输出
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        actor_input = torch.cat((vision_features, current_proprio_obs, vel_output.detach(), feet_height_output.detach(), latent_mean.detach()), dim=-1)
        mean = self.actor(actor_input)
        
        # 6. 记录额外信息
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
        
        # 3. 提取高程图特征 (使用R(2+1)D CNN)
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
        """更新VAE编码器
        
        Args:
            obs_batch: 当前观测批次数据
            next_observations_batch: 下一时刻观测批次数据
            encoder_optimizer: 编码器优化器
            max_grad_norm: 梯度裁剪的最大范数
            
        Returns:
            损失字典，包含各项损失值
        """
        # 更新beta值（单周期退火）
        self.update_beta_schedule()
        
        # 1. 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs_normalized = self.actor_obs_normalizer(policy_obs)
        height_maps_obs = obs_batch["height_scan_policy"]
        # 克隆高程图数据避免影响后续的梯度计算
        sampled_height_maps = height_maps_obs.squeeze(1).clone()
        
        # 2. 提取高程图特征
        vision_features = self.elevation_encoder_actor(sampled_height_maps)
        
        # 3. VAE前向传播
        vel_output, feet_height_output, latent_sample, obs_decode, latent_mean, latent_logvar = \
            self.vae_forward(policy_obs_normalized, vision_features)

        # 4. 获取真实目标值
        # 从critic观测中提取真实速度和脚掌高度
        # Critic观测顺序: base_ang_vel[3] + projected_gravity[3] + torso_ang_vel[3] + torso_projected_gravity[3]
        #                + joint_pos[29] + joint_vel[29] + base_lin_vel[3] + left_foot_height[1] + right_foot_height[1]
        #                + velocity_commands[3] + actions[29]
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs_normalized = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs_normalized[:, 70:73]  # base_lin_vel 在第70-72位置
        feet_height_target = critic_obs_normalized[:, 73:75]  # left_foot_height[73] + right_foot_height[74]

        # 获取下一时刻观测作为重建目标
        next_observations = self.get_critic_obs(next_observations_batch)
        next_observations_normalized = self.critic_obs_normalizer(next_observations)
        obs_target = next_observations_normalized[:, 0:self.num_decode]

        vel_target.requires_grad = False
        feet_height_target.requires_grad = False
        obs_target.requires_grad = False

        # 5. 损失计算
        # 线速度重建损失
        vel_MSE = nn.MSELoss()(vel_output, vel_target)
        # 脚掌高度重建损失
        feet_height_MSE = nn.MSELoss()(feet_height_output, feet_height_target)
        # 观测重建损失
        obs_MSE = nn.MSELoss()(obs_decode, obs_target)
        # KL散度损失（只对隐向量计算）
        dkl_loss = -0.5 * torch.mean(torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1))
        
        # 总损失
        autoenc_loss = vel_MSE + feet_height_MSE + obs_MSE + self.beta * dkl_loss

        # 6. 反向传播
        encoder_optimizer.zero_grad()
        autoenc_loss.backward(retain_graph=True)

        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)

        # 更新参数
        encoder_optimizer.step()

        return {
            "vel_loss": vel_MSE.item(),
            "feet_height_loss": feet_height_MSE.item(),
            "obs_loss": obs_MSE.item(),
            "dkl_loss": dkl_loss.item(),
            "total_loss": autoenc_loss.item(),
            "beta": self.beta,
            "beta_step": self.current_step,
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
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.elevation_encoder_critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        # VAE编码器的优化器（包含actor的高程图编码器）
        encoder_optimizer = optim.Adam([
            {'params': self.elevation_encoder_actor.parameters()},
            {'params': self.vae_encoder.parameters()},
            {'params': self.encoder_vel.parameters()},
            {'params': self.encoder_feet_height.parameters()},
            {'params': self.encoder_latent_mean.parameters()},
            {'params': self.encoder_latent_logvar.parameters()},
            {'params': self.vae_decoder.parameters()},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer,
            "encoder_optimizer": encoder_optimizer
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode11_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode11策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode11_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode11专用的导出器
        exporter = _ElevationNetMode11OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode11OnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode11策略的ONNX导出器"""

    def __init__(self, policy: ActorCriticElevationNetMode11, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 复制策略参数
        if hasattr(policy, "elevation_encoder_actor"):
            self.elevation_encoder = copy.deepcopy(policy.elevation_encoder_actor)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        # 复制VAE相关组件
        if hasattr(policy, "vae_encoder"):
            self.vae_encoder = copy.deepcopy(policy.vae_encoder)
            self.encoder_vel = copy.deepcopy(policy.encoder_vel)
            self.encoder_feet_height = copy.deepcopy(policy.encoder_feet_height)
            self.encoder_latent_mean = copy.deepcopy(policy.encoder_latent_mean)
            self.encoder_latent_logvar = copy.deepcopy(policy.encoder_latent_logvar)
        
        self.elevation_sampled_frames = policy.elevation_sampled_frames
        self.vision_spatial_size = policy.vision_spatial_size
        self.vision_feature_dim = policy.vision_feature_dim
        self.num_latent = policy.num_latent
        # 从actor网络计算本体观测维度：actor输入维度 - 视觉特征维度 - VAE输出维度(3+2+num_latent)
        actor_input_dim = policy.actor[0].in_features  # MLP第一层的输入维度
        self.proprio_obs_dim = actor_input_dim - policy.vision_feature_dim - 3 - 2 - policy.num_latent
        # VAE输入维度：5帧历史本体观测
        self.vae_input_proprio_dim = self.proprio_obs_dim * policy.elevation_sampled_frames

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_input):
        """前向传播（单输入版本）
        
        Args:
            obs_input: 合并的观测数据，形状为 [batch_size, total_obs_dim]
                       前 vae_input_proprio_dim 维是5帧历史本体观测（用于VAE）
                       接下来 proprio_obs_dim 维是当前帧本体观测
                       后面是展平的高程图数据（需要reshape为 [B, sampled_frames, height, width]）
        
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_input.shape[0]
        
        # 切片分离各部分数据
        vae_proprio_data = obs_input[:, :self.vae_input_proprio_dim]  # 5帧历史本体观测
        current_proprio_data = obs_input[:, self.vae_input_proprio_dim:self.vae_input_proprio_dim + self.proprio_obs_dim]  # 当前帧本体观测
        elevation_data_flat = obs_input[:, self.vae_input_proprio_dim + self.proprio_obs_dim:]  # 高程图
        
        # 将高程图数据reshape为 [B, sampled_frames, height, width]
        height, width = self.vision_spatial_size
        elevation_data = elevation_data_flat.reshape(
            batch_size, self.elevation_sampled_frames, height, width
        )
        
        # 应用归一化器
        vae_proprio_data = self.normalizer(vae_proprio_data)
        current_proprio_data = self.normalizer(current_proprio_data)
        
        # 提取视觉特征 (使用R(2+1)D CNN)
        vision_features = self.elevation_encoder(elevation_data)
        
        # VAE前向传播
        vae_input = torch.cat([vae_proprio_data, vision_features.detach()], dim=-1)
        x = self.vae_encoder(vae_input)
        vel_output = self.encoder_vel(x)
        feet_height_output = self.encoder_feet_height(x)
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        latent_logvar = torch.clamp(latent_logvar, min=-10, max=10)  # 使用 clamp 而非 clip
        
        # 重参数化采样
        std = torch.exp(latent_logvar * 0.5)
        latent_sample = latent_mean + std * torch.randn_like(std)
        
        # 融合所有特征
        fused_features = torch.cat([vision_features, current_proprio_data, vel_output, feet_height_output, latent_sample], dim=-1)
        
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
        total_obs_dim = self.vae_input_proprio_dim + self.proprio_obs_dim + elevation_dim
        
        # 创建单个合并的输入示例
        # 格式: [5帧历史本体观测 + 当前帧本体观测 + 展平的高程图]
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (Mode11单输入模式 - R(2+1)D CNN + VAE):")
        print(f"{'='*80}")
        print(f"  5帧历史本体观测维度:   {self.vae_input_proprio_dim}")
        print(f"  当前帧本体观测维度:     {self.proprio_obs_dim}")
        print(f"  高程图维度:             {elevation_dim} ({sampled_frames}×{height}×{width})")
        print(f"  VAE输出维度:            {3 + 2 + self.num_latent} (速度3 + 足端2 + 隐向量{self.num_latent})")
        print(f"  总输入维度:             {total_obs_dim}")
        print(f"  输入切片方式:")
        print(f"    [:, :{self.vae_input_proprio_dim}] = 5帧历史本体")
        print(f"    [:, {self.vae_input_proprio_dim}:{self.vae_input_proprio_dim + self.proprio_obs_dim}] = 当前本体")
        print(f"    [:, {self.vae_input_proprio_dim + self.proprio_obs_dim}:] = 高程图")
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
