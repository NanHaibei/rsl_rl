# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""ActorCriticElevationNetMode10: VAE编码器架构，使用R(2+1)D CNN处理高程图序列

网络结构:
    本体 -> 本体编码器MLP -> 特征1
    高程图序列 -> R(2+1)D CNN编码器(分解时空卷积) -> 特征2
    [特征1 + 特征2] -> 融合MLP -> 编码特征
    编码特征 -> Encoder -> 隐向量(v+z)
    隐向量 -> Decoder -> 重建观测
    [隐向量 + 本体] -> Actor MLP -> 动作
    
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
import numpy as np

from rsl_rl import env
from rsl_rl.networks import MLP, EmpiricalNormalization
import copy
import os


class AdaBootManager:
    """AdaBoot (Adaptive Bootstrapping) 管理器
    
    根据训练稳定性自适应调整bootstrap概率：
    - 公式: p_boot = 1 - tanh(CV(R) * scale)
    - CV = std(rewards) / |mean(rewards)|
    - 早期训练（高方差）→ 低p_boot → 少用estimator
    - 后期训练（低方差）→ 高p_boot → 多用estimator增强鲁棒性
    
    Attributes:
        warmup_iterations: 预热迭代次数
        episodic_rewards_buffer: episode奖励缓冲区
    """
    
    def __init__(self, config: dict | None = None, env_cfg=None) -> None:
        """初始化AdaBoot管理器"""

        self.warmup_iterations = config["warmup_iterations"]
        self.cv_scale = config["cv_scale"]
        self.type = config["type"]
        self.current_iteration = 0  # 当前训练迭代次数
        self.probability = 0.0  # 当前bootstrap概率
        
        # 区分三种不同的boot模式
        if self.type == "CV":
            self.episodic_rewards_buffer = torch.zeros(config["num_envs"], dtype=torch.float32, device=config["device"])
            self.completed_episodes_rewards = []             # 存储本次rollout中完成的episodes的总奖励
        elif self.type == "terrain":
            self.now_terrain_level = 0.0
            self.max_terrain_level = config["max_terrain_level"]
        elif self.type == "xyVel":
            self.now_tracking_reward = 0.0
            self.max_tracking_reward = env_cfg.rewards.track_lin_vel_xy_exp.weight * config["max_tracking_ratio"]
        else:
            raise ValueError(f"Unknown AdaBoot type: {self.type}")
        
        print(f"[AdaBoot] 已启用 - 预热迭代: {self.warmup_iterations}, CV缩放: {self.cv_scale}")
    
    def update_data(self, rewards: torch.Tensor, dones: torch.Tensor, extras) -> None:
        """更新episode奖励（在每个环境step时调用）
        
        Args:
            rewards: 当前步的奖励 [num_envs]
            dones: 环境是否完成 [num_envs]
        """
        
        if self.type == "CV":
            # 累加所有环境的奖励
            self.episodic_rewards_buffer += rewards
            
            # 对于done的环境，保存完整的episode奖励到列表（向量化操作）
            if dones.any():
                done_rewards = self.episodic_rewards_buffer[dones].tolist()
                self.completed_episodes_rewards.extend(done_rewards)
                
                # 重置done环境的累积奖励
                self.episodic_rewards_buffer[dones] = 0.0
        elif self.type == "terrain":
            self.now_terrain_level = extras["log"]["Curriculum/terrain_levels"]
        elif self.type == "xyVel":
            self.now_tracking_reward = extras["log"]["Episode_Reward/track_lin_vel_xy_exp"]

    
    def update_iteration(self) -> None:
        """在每个训练迭代结束时调用，更新bootstrap概率
        
        基于本次rollout中完成的episodes计算CV和p_boot
        """
        
        self.current_iteration += 1
        # 预热期：强制不使用bootstrap
        if self.current_iteration <= self.warmup_iterations:
            self.probability = 0.0
            if self.type == "CV":
                self.completed_episodes_rewards.clear()
            return
            
        if self.type == "CV":
            # 需要至少有一些完成的episodes才能计算CV
            if len(self.completed_episodes_rewards) < 10:
                self.probability = 0.0
                return
            
            # 计算CV (Coefficient of Variation)
            rewards_array = torch.tensor(self.completed_episodes_rewards, dtype=torch.float32)
            mean_reward = torch.mean(rewards_array).item()
            std_reward = torch.std(rewards_array, unbiased=False).item()
            
            # 避免除以零
            cv = std_reward / (abs(mean_reward) + 1e-6)
            
            # 计算bootstrap概率: probability = 1 - tanh(CV * scale)
            self.probability = float(torch.clip(
                1.0 - torch.tanh(torch.tensor(cv * self.cv_scale)), 0.0, 1.0).item())
            
            # 清空本次rollout的统计数据
            self.completed_episodes_rewards.clear()
        elif self.type == "terrain":
            self.probability = float(torch.clip(
                torch.tensor(self.now_terrain_level / self.max_terrain_level), 0.0, 1.0).item())
        elif self.type == "xyVel":
            self.probability = float(torch.clip(
                torch.tensor(self.now_tracking_reward / self.max_tracking_reward), 0.0, 1.0).item())
    
    def should_use_estimate(self) -> bool:
        """决定是否使用估计值（bootstrap）
        
        根据当前概率随机采样决定
        
        Returns:
            True表示使用估计值，False表示使用真实值
        """
        
        # 根据当前概率随机采样
        return torch.rand(1).item() < self.probability
    
    def get_stats(self) -> dict[str, float]:
        """获取AdaBoot统计信息
        
        Returns:
            包含统计信息的字典
        """
        
        stats = {"adaboot/probability": self.probability}
        
        # 根据不同模式添加特定的统计信息
        if self.type == "CV":
            # 计算当前缓存的episodes统计（如果有）
            if len(self.completed_episodes_rewards) > 0:
                rewards_array = torch.tensor(self.completed_episodes_rewards, dtype=torch.float32)
                mean_r = torch.mean(rewards_array).item()
                std_r = torch.std(rewards_array, unbiased=False).item()
                cv = std_r / (abs(mean_r) + 1e-6)
            else:
                mean_r = std_r = cv = 0.0
            
            stats.update({
                "adaboot/cv": cv,
                "adaboot/mean_reward": mean_r,
                "adaboot/std_reward": std_r,
                "adaboot/num_completed_episodes": float(len(self.completed_episodes_rewards))
            })
        return stats

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
        # 归一化
        x_mean = x.mean(dim=(-1, -2), keepdim=True)
        x = torch.clip((x - x_mean) / 0.1, -5.0, 5.0)
        
        # R(2+1)D卷积
        x = self._forward_conv(x)
        
        # 展平并全连接
        x = x.flatten(1)
        x = self.fc(x)
        return x

class ActorCriticElevationNetMode10(nn.Module):
    """Mode10: 使用R(2+1)D CNN处理高程图序列（分解时空卷积）+ VAE架构
    
    网络组成:
    1. 本体编码器MLP - 提取本体特征
    2. 高程图R(2+1)D CNN编码器 - 通过分解的2D空间卷积和1D时间卷积提取时空特征
    3. 融合MLP - 融合特征
    4. VAE Encoder - 输出隐向量(速度v + 隐状态z)
    5. VAE Decoder - 重建观测
    6. Actor MLP - 从隐向量+本体观测输出动作
    7. Critic MLP - 价值评估
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
        history_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (25, 17),
        conv2d_hidden_dims: list[int] = [16, 32, 64],
        conv2d_kernel_sizes: list[int] = [3, 3, 3],
        vision_feature_dim: int = 64,
        # 本体编码器配置
        proprio_feature_dim: int = 128,
        proprio_encoder_hidden_dims: list[int] | None = None,
        # VAE编码器-解码器配置
        encoder_hidden_dims: tuple[int] | list[int] = [1024, 512, 256],
        decoder_hidden_dims: tuple[int] | list[int] = [256, 512, 1024],
        num_encoder_output:int = 32,
        num_latent: int = 14,
        num_elevation_latent: int = 16,
        num_decode: int = 70,
        VAE_beta: float = 1.0,
        use_beta_annealing: bool = True,
        beta_start: float = 0.01,
        beta_end: float = 0.5,
        beta_anneal_steps: int = 10000,
        adaboot_cfg: dict | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        # 配置
        self.extra_info = dict()
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size
        self.noise_std_type = noise_std_type
        self.history_frames = history_frames
        self.vision_feature_dim = vision_feature_dim
        
        # Beta配置：根据标志位决定使用退火还是固定值
        self.use_beta_annealing = use_beta_annealing
        if self.use_beta_annealing:
            # 从alg_cfg中获取训练配置（如果有的话）
            if alg_cfg is not None:
                self.num_learning_epochs = alg_cfg.get('num_learning_epochs', 5)
                self.num_mini_batches = alg_cfg.get('num_mini_batches', 4)
            else:
                # 使用默认值
                self.num_learning_epochs = 5
                self.num_mini_batches = 4
            
            self.updates_per_iteration = self.num_learning_epochs * self.num_mini_batches
            
            # 使用退火策略
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_anneal_steps = beta_anneal_steps
            self.beta = self.beta_start  # 初始化为起始值
            self.current_step = 0
            self.update_call_count = 0  # 记录update_beta_schedule被调用的次数
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
        
        # 通过 vision_spatial_size 计算高程图重建维度
        self.num_elevation_decode = vision_spatial_size[0] * vision_spatial_size[1]
        self.num_decode = num_decode
        self.num_encoder_output = num_encoder_output
        self.num_latent = num_latent
        self.num_elevation_latent = num_elevation_latent
        self.total_latent_dim = 3 + 2 + num_latent + num_elevation_latent  # 速度(3) + 脚掌高度(2) + 纯隐变量 + 高程图隐变量
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / history_frames)
        
        ########################################## 网络架构 ##############################################
        # 1. Actor网络
        self.actor = MLP(self.num_proprio_one_frame + self.total_latent_dim, num_actions, actor_hidden_dims, activation)
        
        # 2. Critic网络
        self.elevation_encoder_critic = R2Plus1DElevationEncoder(
            num_frames=history_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.critic = MLP(num_critic_obs + vision_feature_dim, 1, critic_hidden_dims, activation)

        # 3. encoder-特征提取
        self.elevation_feature = R2Plus1DElevationEncoder(
            num_frames=history_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.proprio_feature = MLP(num_actor_obs, proprio_feature_dim, proprio_encoder_hidden_dims, activation)
        
        # 4. encoder-VAE：输出4个部分
        self.encoder = MLP(proprio_feature_dim + vision_feature_dim, num_encoder_output, encoder_hidden_dims, activation)
        # 速度编码器 (3维) - 直接输出，无需重参数化
        self.encoder_vel = nn.Linear(num_encoder_output, 3)
        # 脚掌高度编码器 (2维) - 直接输出，无需重参数化
        self.encoder_feet_height = nn.Linear(num_encoder_output, 2)
        # 纯隐变量编码器
        self.encoder_latent_mean = nn.Linear(num_encoder_output, num_latent)
        self.encoder_latent_logvar = nn.Linear(num_encoder_output, num_latent)
        # 高程图隐变量编码器
        self.encoder_elevation_latent_mean = nn.Linear(num_encoder_output, num_elevation_latent)
        self.encoder_elevation_latent_logvar = nn.Linear(num_encoder_output, num_elevation_latent)

        # 5. 主decoder：重建观测 (使用所有隐变量)
        self.decoder = MLP(self.total_latent_dim, num_decode, decoder_hidden_dims, activation)
        
        # 6. 高程图decoder：重建无噪声高程图 (只使用高程图隐变量)
        self.elevation_decoder = MLP(num_elevation_latent, self.num_elevation_decode, decoder_hidden_dims, activation)

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
        
        # AdaBoot管理器
        if adaboot_cfg is not None:
            num_envs = obs.shape[0]
            device = obs["policy"].device
            adaboot_cfg["num_envs"] = num_envs
            adaboot_cfg["device"] = device
            self.adaboot_manager = AdaBootManager(adaboot_cfg, env_cfg=env_cfg) 
        else:
            self.adaboot_manager = None
        
        # 打印网络结构
        print("\n" + "="*80)
        print("ActorCriticElevationNetMode10 网络结构")
        print("="*80)
        print(self)
        print("="*80 + "\n")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError
    
    def update_beta_schedule(self) -> None:
        """更新beta值的退火调度
        
        此方法会在每个mini batch的encoder update时被调用，但内部会计数，
        只有当调用次数达到 num_learning_epochs * num_mini_batches 时才真正更新beta值。
        
        单周期退火：从beta_start线性增长到beta_end，然后保持不变
        - 0 ~ beta_anneal_steps: 线性增长
        - > beta_anneal_steps: 保持在beta_end
        
        如果use_beta_annealing=False，则beta保持固定值不变
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
    
    def encoder_forward(self, proprio_obs: torch.Tensor, height_maps_obs: TensorDict):
        """编码器前向传播
        
        Returns:
            code: 完整隐向量 [速度(3) + 脚掌高度(2) + 纯隐变量 + 高程图隐变量]
            vel_output: 速度直接输出值
            feet_height_output: 脚掌高度直接输出值
            latent_sample: 纯隐变量采样值
            elevation_latent_sample: 高程图隐变量采样值
            obs_decode: 主decoder重建的观测
            elevation_decode: 高程图decoder重建的无噪声高程图
            latent_mean, latent_logvar: 纯隐变量的均值和方差
            elevation_latent_mean, elevation_latent_logvar: 高程图隐变量的均值和方差
        """
        # 1. 提取特征
        proprio_features = self.proprio_feature(proprio_obs)
        vision_features = self.elevation_feature(height_maps_obs)
        
        # 2. 融合特征并通过encoder MLP
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        x = self.encoder(fused_features)
        
        # 3. VAE编码: 输出4个部分
        # 速度 (3维) - 直接输出
        vel_output = self.encoder_vel(x)
        # 脚掌高度 (2维) - 直接输出
        feet_height_output = self.encoder_feet_height(x)
        # 纯隐变量
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        # 高程图隐变量
        elevation_latent_mean = self.encoder_elevation_latent_mean(x)
        elevation_latent_logvar = self.encoder_elevation_latent_logvar(x)
        
        # 限制方差范围
        latent_logvar = torch.clip(latent_logvar, min=-10, max=10)
        elevation_latent_logvar = torch.clip(elevation_latent_logvar, min=-10, max=10)
        
        # 4. 采样隐向量（只对纯隐变量和高程图隐变量）
        latent_sample = self.reparameterise(latent_mean, latent_logvar)
        elevation_latent_sample = self.reparameterise(elevation_latent_mean, elevation_latent_logvar)
        
        # 5. 拼接成完整隐向量
        code = torch.cat((vel_output, feet_height_output, latent_sample, elevation_latent_sample), dim=-1)
        
        # 6. 主decoder：重建观测 (使用所有隐变量)
        obs_decode = self.decoder(code)
        
        # 7. 高程图decoder：重建无噪声高程图 (只使用高程图隐变量)
        elevation_decode = self.elevation_decoder(elevation_latent_sample)
        
        return (code, vel_output, feet_height_output, latent_sample, elevation_latent_sample,
                obs_decode, elevation_decode,
                latent_mean, latent_logvar, elevation_latent_mean, elevation_latent_logvar)

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样
        
        支持AdaBoot：
        - 根据AdaBoot管理器的概率决定是否使用estimator的估计值
        - 概率高时更多使用估计值（bootstrap）
        - 概率低时更多使用真实值（ground truth）
        """
        # 1. 获取并归一化本体观测 获取高程图观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        height_maps_obs = obs["height_scan_policy"]
        
        # 2. 编码器前向传播得到隐向量
        (code, vel_output, feet_height_output, latent_sample, elevation_latent_sample,
         obs_decode, elevation_decode,
         latent_mean, latent_logvar, elevation_latent_mean, elevation_latent_logvar) = \
            self.encoder_forward(proprio_obs_normalized, height_maps_obs)
        
        # 3. AdaBoot逻辑：根据管理器决定使用估计值还是真实值
        now_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]  # 当前观测
        
        # 由AdaBoot管理器决定是否使用估计值
        use_estimate = True if self.adaboot_manager is None else self.adaboot_manager.should_use_estimate()
        
        if use_estimate:
            # 使用estimator的估计值（训练estimator）
            observation = torch.cat((code.detach(), now_obs), dim=-1)
        else:
            # 使用真实值（从critic_obs获取）
            critic_obs = self.get_critic_obs(obs)
            critic_obs_normalized = self.critic_obs_normalizer(critic_obs)
            real_velocity = critic_obs_normalized[:, 70:73]  # 提取真实速度
            real_feet_height = critic_obs_normalized[:, 73:75]  # 提取真实脚掌高度
            # 拼接真实值、隐变量采样值和当前观测
            real_code = torch.cat((real_velocity.detach(), real_feet_height.detach(), 
                                   latent_sample.detach(), elevation_latent_sample.detach()), dim=-1)
            observation = torch.cat((real_code, now_obs), dim=-1)
        
        # 4. Actor输出动作
        mean = self.actor(observation)
        self._update_distribution(mean)
        
        # 5. 记录额外信息用于监控
        self.extra_info["est_vel"] = vel_output
        self.extra_info["est_feet_height"] = feet_height_output
        
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = obs_decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                                       self.actor_obs_normalizer.mean[:self.num_decode]
            # 高程图重建不需要反归一化，因为高程图有自己的归一化
            self.extra_info["elevation_predict"] = elevation_decode
        else:
            self.extra_info["obs_predict"] = obs_decode
            self.extra_info["elevation_predict"] = elevation_decode
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 获取并归一化本体观测 获取高程图观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        height_maps_obs = obs["height_scan_policy"]
        
        # 2. 编码器前向传播得到隐向量
        (code, vel_output, feet_height_output, latent_sample, elevation_latent_sample,
         obs_decode, elevation_decode,
         latent_mean, latent_logvar, elevation_latent_mean, elevation_latent_logvar) = \
            self.encoder_forward(proprio_obs, height_maps_obs)
        
        # 3. 推理时使用均值而非采样值
        now_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 取当前观测值部分
        observation = torch.cat((vel_output.detach(), feet_height_output.detach(), 
                                latent_mean.detach(), elevation_latent_mean.detach(), now_obs), dim=-1)
        
        # 4. Actor输出确定性动作
        mean = self.actor(observation)
        
        # 5. 记录额外信息
        self.extra_info["est_vel"] = vel_output
        self.extra_info["est_feet_height"] = feet_height_output
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = obs_decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                            self.actor_obs_normalizer.mean[:self.num_decode]
            self.extra_info["elevation_predict"] = elevation_decode
        else:
            self.extra_info["obs_predict"] = obs_decode
            self.extra_info["elevation_predict"] = elevation_decode
        
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值"""
        # 1. 提取观测值
        height_maps_obs = obs["height_scan_critic"]
        current_critic_obs = self.get_critic_obs(obs)
        
        # 应用观测归一化
        current_critic_obs = self.critic_obs_normalizer(current_critic_obs)
        
        # 2. 提取高程图特征
        vision_features = self.elevation_encoder_critic(height_maps_obs)
        
        # 3. 融合特征
        fused_features = torch.cat((vision_features, current_critic_obs), dim=-1)
        
        # 4. Critic输出价值
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
            if obs_group not in ["height_scan_history", "height_scan_policy"]:
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
        
        # 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        height_maps_obs = obs_batch["height_scan_policy"]

        # 前向传播得到编码器输出
        (code, vel_output, feet_height_output, latent_sample, elevation_latent_sample,
         obs_decode, elevation_decode,
         latent_mean, latent_logvar, elevation_latent_mean, elevation_latent_logvar) = \
            self.encoder_forward(policy_obs, height_maps_obs)

        # 获取并归一化critic观测，提取真实速度和脚掌高度
        # Critic观测顺序: base_ang_vel[3] + projected_gravity[3] + torso_ang_vel[3] + torso_projected_gravity[3]
        #                + joint_pos[29] + joint_vel[29] + base_lin_vel[3] + left_foot_height[1] + right_foot_height[1]
        #                + velocity_commands[3] + actions[29]
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs[:, 70:73]  # base_lin_vel 在第70-72位置
        feet_height_target = critic_obs[:, 73:75]  # left_foot_height[73] + right_foot_height[74]

        # 获取下一时刻观测，提取目标观测
        next_observations = self.get_critic_obs(next_observations_batch)
        next_observations = self.critic_obs_normalizer(next_observations)
        obs_target = next_observations[:, 0:self.num_decode]  # 取最新观测
        
        # 获取无噪声高程图作为目标（从height_scan_critic获取，因为它没有噪声）
        elevation_target = obs_batch["height_scan_critic"][:, 0, :, :].flatten(1)  # 取当前帧并展平

        vel_target.requires_grad = False
        feet_height_target.requires_grad = False
        obs_target.requires_grad = False
        elevation_target.requires_grad = False

        # 损失计算：
        # 1. 速度重建损失（直接输出，无需采样）
        vel_MSE = nn.MSELoss()(vel_output, vel_target)
        # 2. 脚掌高度重建损失（直接输出，无需采样）
        feet_height_MSE = nn.MSELoss()(feet_height_output, feet_height_target)
        # 3. 观测重建损失
        obs_MSE = nn.MSELoss()(obs_decode, obs_target)
        # 4. 高程图重建损失
        elevation_MSE = nn.MSELoss()(elevation_decode, elevation_target)
        # 5. KL散度损失（只对纯隐变量和高程图隐变量计算，不对线速度和足端高度计算）
        dkl_latent = -0.5 * torch.mean(torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1))
        dkl_elevation = -0.5 * torch.mean(torch.sum(1 + elevation_latent_logvar - elevation_latent_mean.pow(2) - elevation_latent_logvar.exp(), dim=1))
        dkl_loss = dkl_latent + dkl_elevation
        
        # 总损失
        autoenc_loss = vel_MSE + feet_height_MSE + obs_MSE + elevation_MSE + self.beta * dkl_loss

        # 反向传播
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
            "elevation_loss": elevation_MSE.item(),
            "dkl_latent": dkl_latent.item(),
            "dkl_elevation": dkl_elevation.item(),
            "dkl_loss": dkl_loss.item(),
            "total_loss": autoenc_loss.item(),
            "beta": self.beta,
            "beta_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典，包含主要的优化器和编码器优化器
        """
        import torch.optim as optim
        
        optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        encoder_optimizer = optim.Adam([
            {'params': self.proprio_feature.parameters()},
            {'params': self.elevation_feature.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.encoder_vel.parameters()},
            {'params': self.encoder_feet_height.parameters()},
            {'params': self.encoder_latent_mean.parameters()},
            {'params': self.encoder_latent_logvar.parameters()},
            {'params': self.encoder_elevation_latent_mean.parameters()},
            {'params': self.encoder_elevation_latent_logvar.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.elevation_decoder.parameters()},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer,
            "encoder_optimizer": encoder_optimizer
        }

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode10_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode10策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode10_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode10专用的导出器
        exporter = _ElevationNetMode10OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode10OnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode10策略的ONNX导出器"""

    def __init__(self, policy: ActorCriticElevationNetMode10, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 复制策略参数
        if hasattr(policy, "proprio_feature"):
            self.proprio_feature = copy.deepcopy(policy.proprio_feature)
        if hasattr(policy, "elevation_feature"):
            self.elevation_feature = copy.deepcopy(policy.elevation_feature)
        if hasattr(policy, "encoder"):
            self.encoder = copy.deepcopy(policy.encoder)
        if hasattr(policy, "encoder_vel"):
            self.encoder_vel = copy.deepcopy(policy.encoder_vel)
        if hasattr(policy, "encoder_feet_height"):
            self.encoder_feet_height = copy.deepcopy(policy.encoder_feet_height)
        if hasattr(policy, "encoder_latent_mean"):
            self.encoder_latent_mean = copy.deepcopy(policy.encoder_latent_mean)
        if hasattr(policy, "encoder_elevation_latent_mean"):
            self.encoder_elevation_latent_mean = copy.deepcopy(policy.encoder_elevation_latent_mean)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        self.num_proprio_one_frame = policy.num_proprio_one_frame
        self.vision_spatial_size = policy.vision_spatial_size

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        # 输入需要包含高程图序列数据，这里简化处理
        # 实际使用时需要根据具体输入格式调整
        obs_len = self.normalizer.in_features
        proprio_obs = x[:, 0:obs_len]
        height_data = x[:, obs_len:]  # 包含高程图序列数据
        
        # 归一化本体观测
        normalized_obs = self.normalizer(proprio_obs)
        
        # 重塑高程图数据（假设输入是展平的）
        # 这里需要根据实际输入格式调整
        batch_size = x.shape[0]
        height, width = self.vision_spatial_size
        sequence_length = height_data.shape[1] // (height * width)
        height_map_sequence = height_data.view(batch_size, sequence_length, height, width)
        
        # 提取特征
        proprio_features = self.proprio_feature(normalized_obs)
        vision_features = self.elevation_feature(height_map_sequence)
        
        # 融合特征
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        x = self.encoder(fused_features)
        
        # VAE编码：使用均值（推理模式）
        vel_output = self.encoder_vel(x)
        feet_height_output = self.encoder_feet_height(x)
        latent_mean = self.encoder_latent_mean(x)
        elevation_latent_mean = self.encoder_elevation_latent_mean(x)
        
        # 拼接隐向量
        code = torch.cat((vel_output, feet_height_output, latent_mean, elevation_latent_mean), dim=-1)
        
        # 与当前本体观测拼接
        now_obs = normalized_obs[:, 0:self.num_proprio_one_frame]
        observation = torch.cat((code.detach(), now_obs), dim=-1)
        
        # 输出动作
        actions_mean = self.actor(observation)
        return actions_mean, vel_output, feet_height_output

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        # 创建输入示例（简化版本）
        # 实际使用时需要根据高程图的实际尺寸计算
        height, width = self.vision_spatial_size
        height_map_dim = height * width * 5  # 假设5帧
        total_dim = self.normalizer.in_features + height_map_dim
        obs = torch.zeros(1, total_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions", "est_vel", "est_feet_height"],
            dynamic_axes={},
        )
