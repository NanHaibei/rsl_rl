# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode12P2: VAE架构的编码器-解码器网络

网络结构（修正版）:

    Critic Pipeline (PPO):
        本体观测值（单帧，例如G1中107维）-> MLP -> 特征向量
        高程图历史值（5帧，例如G1中5×425=2125）-> 2+1D CNN -> 特征向量
        [两个特征concat] -> MLP Critic -> V
    
    Encoder-Decoder Pipeline (VAE架构):
        高程图编码器 (VAE):
            高程图历史值（critic的高程图，5帧无噪声）-> 2+1D CNN -> 
                -> Linear -> μ_t^e, σ_t^e (VAE隐变量)
                -> 重参数化 -> z_t^e (高程图隐向量, 默认32维)
        
        高程图解码器:
            z_t^e -> MLP decoder -> 重建critic obs中的无噪声高程图（例如2125维）
        
        本体编码器:
            本体历史观测值（policy的多帧观测，例如G1中96×5=480维）-> MLP encoder ->
                -> Linear -> v̂_t (线速度估计值, 3维)
                -> Linear -> z_t^p (本体信息隐向量, 默认32维)
        
        本体解码器:
            [z_t^e + z_t^p + v̂_t] -> MLP -> 重建下一帧本体信息（num_decode维）
    
    Actor Pipeline:
        [单帧本体观测值（例如G1中96维）+ v̂_t + z_t^e + z_t^p] -> MLP Actor -> a_t
        
观测组说明（以G1为例）:
    - policy: 单帧本体观测96维，环境会复制5帧形成历史（96×5=480）
    - critic: 单帧完整状态107维（包含额外的torso信息、足端高度等）
    - height_scan_policy: 高程图5帧历史，每帧25×17=425（共2125维）
    - height_scan_critic: 无噪声高程图5帧历史（共2125维）
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
        self.vision_spatial_size = vision_spatial_size
        
        # 构建R(2+1)D卷积层
        layers = []
        in_channels = 1  # 单通道高程图
        
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
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
        x = torch.clip((x - x_mean) / 0.6, -3.0, 3.0)
        
        # 通过R(2+1)D卷积
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ActorCriticElevationNetMode12P2(nn.Module):
    """
    完整的Actor-Critic网络，包含VAE编码器-解码器
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
        # 编码器参数（Mode12P2特有）
        encoder_hidden_dims: list[int] = [256, 256],
        latent_dim: int = 32,  # 高程图VAE隐变量维度
        vel_dim: int = 3,      # 速度维度
        proprio_latent_dim: int = 32,   # 本体隐变量维度
        # 解码器参数（Mode12P2特有）
        decoder_hidden_dims: list[int] = [256, 256],
        num_decode: int = 51,  # 重建的观测维度
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
        self.latent_dim = latent_dim
        self.vel_dim = vel_dim
        self.proprio_latent_dim = proprio_latent_dim
        self.num_decode = num_decode
        # 自动计算高程图解码大小
        self.elevation_decode_size = elevation_sampled_frames * vision_spatial_size[0] * vision_spatial_size[1]
        
        # 计算观测维度
        # policy: 包含5帧历史的本体观测（已拼接）
        # 例如：G1中单帧policy观测为96维，5帧历史后num_actor_obs=96*5=480
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        # critic: 单帧完整状态观测（不包含高程图）
        # 例如：G1中critic观测为107维（包含torso信息、足端高度等额外信息）
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        # 单帧本体观测维度 = 多帧观测 / 帧数
        # 例如：G1中 num_proprio_one_frame = 480 // 5 = 96
        self.num_proprio_one_frame = num_actor_obs // elevation_sampled_frames
        
        # activation用于Sequential，activation_name用于MLP
        activation_class = nn.ELU if activation == "elu" else nn.ReLU
        activation_name = activation  # 字符串形式
        
        # ============ Critic部分（PPO） ============
        
        # Critic的MLP特征提取器（处理本体观测）
        self.critic_mlp_feature_extractor = MLP(
            num_critic_obs,
            critic_mlp_feature_dim,
            mlp_extractor_hidden_dims,
            activation
        )
        
        # Critic的高程图编码器（处理高程图历史）
        self.critic_elevation_encoder = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size,
        )
        
        # Critic网络：[MLP特征 + 高程图特征] -> Value
        self.critic = MLP(
            critic_mlp_feature_dim + vision_feature_dim,
            1,
            critic_hidden_dims,
            activation
        )
        
        # ============ 编码器部分（VAE架构） ============
        
        # 1. 高程图VAE编码器：高程图 -> VAE隐变量 (μ, σ) -> z_e
        self.elevation_vae_encoder = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size,
        )
        
        # 高程图VAE的均值和方差head
        self.elevation_mu = nn.Linear(vision_feature_dim, latent_dim)
        self.elevation_logvar = nn.Linear(vision_feature_dim, latent_dim)
        
        # 2. 本体编码器：本体历史 -> [v̂_t, z_p]
        # num_actor_obs已经是5帧拼接后的观测，不需要再乘以elevation_sampled_frames
        # 例如：G1中 num_actor_obs = 96*5 = 480
        self.proprio_encoder = MLP(
            input_dim=num_actor_obs,
            hidden_dims=encoder_hidden_dims,
            output_dim=encoder_hidden_dims[-1],
            activation=activation_name,
        )
        
        # 本体编码器的输出head（输入包含proprio_features + mu_e）
        self.proprio_vel_head = nn.Linear(encoder_hidden_dims[-1] + latent_dim, vel_dim)  # 速度估计
        self.proprio_latent_head = nn.Linear(encoder_hidden_dims[-1] + latent_dim, proprio_latent_dim)  # 本体隐变量
        
        # ============ KL权重配置 ============
        self.kl_weight = kwargs.get('kl_weight', 0.001)  # 从配置读取，默认0.001
        
        # ============ 解码器部分 ============
        
        # 3. 高程图解码器：z_e -> 重建critic obs中的无噪声高程图
        self.elevation_decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.elevation_decode_size,
            activation=activation_name,
        )
        
        # 4. 本体解码器：[z_e + z_p + v̂] -> 重建下一帧本体信息
        decoder_input_dim = latent_dim + proprio_latent_dim + vel_dim
        self.obs_decoder = MLP(
            input_dim=decoder_input_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=num_decode,
            activation=activation_name,
        )
        
        # ============ Actor部分 ============
        
        # Actor网络：[单帧本体观测 + v̂ + z_e + z_p] -> 动作
        # 单帧本体观测维度：num_proprio_one_frame（例如G1中为96）
        # v̂维度：vel_dim（速度估计，通常为3）
        # z_e维度：latent_dim（高程图VAE隐变量，默认32）
        # z_p维度：proprio_latent_dim（本体隐变量，默认32）
        actor_input_dim = self.num_proprio_one_frame + vel_dim + latent_dim + proprio_latent_dim
        self.actor = MLP(
            input_dim=actor_input_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_name,
        )
        
        # ============ 归一化层 ============
        # Actor observation normalization（针对完整5帧历史观测）
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

        # ============ 动作分布的标准差 ============
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
        print("ActorCriticElevationNetMode12P2 网络结构")
        print("="*80)
        print(self)
        print("="*80 + "\n")
    
    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass
        
    def forward(self) -> NoReturn:
        raise NotImplementedError("Use act() or act_inference() instead.")
    
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
    
    # ============ PPO相关方法 ============
    
    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        proprio_obs = obs["policy"]  # 已包含5帧历史拼接（例如G1中480维）
        
        # 对整个5帧历史进行归一化
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        
        # 提取归一化后的单帧观测
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1) if height_maps.dim() > 4 else height_maps
        
        # 3. 编码器：获取隐变量（使用无梯度推理）
        with torch.no_grad():
            # 高程图编码 -> z_e
            elevation_features = self.elevation_vae_encoder(sampled_height_maps)
            mu_e = self.elevation_mu(elevation_features)
            
            # 本体编码 -> v̂, z_p（使用归一化后的完整5帧历史 + mu_e）
            proprio_features = self.proprio_encoder(proprio_obs_normalized)
            proprio_mu_e_features = torch.cat([proprio_features, mu_e], dim=-1)
            v_hat = self.proprio_vel_head(proprio_mu_e_features)
            z_p = self.proprio_latent_head(proprio_mu_e_features)
        
        # 4. 融合特征
        fused_features = torch.cat([current_proprio_obs, v_hat, mu_e, z_p], dim=-1)
        
        # 5. Actor输出动作
        mean = self.actor(fused_features)
        self._update_distribution(mean)
        
        return self.distribution.sample(), self.extra_info
    
    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 提取观测值
        height_maps = obs["height_scan_policy"]
        proprio_obs = obs["policy"]  # 已包含5帧历史拼接
        
        # 对整个5帧历史进行归一化
        proprio_obs_normalized = self.actor_obs_normalizer(proprio_obs)
        
        # 提取归一化后的单帧观测
        current_proprio_obs = proprio_obs_normalized[:, 0:self.num_proprio_one_frame]
        
        # 2. 整理高程图格式
        sampled_height_maps = height_maps.squeeze(1) if height_maps.dim() > 4 else height_maps
        
        # 3. 编码器：获取隐变量
        print(obs["height_scan_critic"][0,0,:])
        elevation_features = self.elevation_vae_encoder(sampled_height_maps)
        mu_e = self.elevation_mu(elevation_features)
        
        # 本体编码（使用归一化后的完整5帧历史 + mu_e）
        proprio_features = self.proprio_encoder(proprio_obs_normalized)
        proprio_mu_e_features = torch.cat([proprio_features, mu_e], dim=-1)
        v_hat = self.proprio_vel_head(proprio_mu_e_features)
        z_p = self.proprio_latent_head(proprio_mu_e_features)
        
        # 4. 融合特征
        fused_features = torch.cat([current_proprio_obs, v_hat, mu_e, z_p], dim=-1)
        
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
        sampled_height_maps = height_maps.squeeze(1) if height_maps.dim() > 4 else height_maps
        
        # 3. 提取MLP特征
        mlp_features = self.critic_mlp_feature_extractor(current_proprio_obs)
        
        # 4. 提取高程图特征
        vision_features = self.critic_elevation_encoder(sampled_height_maps)
        
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
    
    # ============ VAE编码器-解码器方法 ============
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE重参数化技巧
        
        Args:
            mu: 均值 [B, latent_dim]
            logvar: 对数方差 [B, latent_dim]
            
        Returns:
            z: 采样的隐变量 [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def update_encoder(
        self,
        obs_batch: TensorDict,
        next_obs_batch: TensorDict,
        encoder_optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """更新编码器-解码器
        
        Args:
            obs_batch: 当前观测batch
            next_obs_batch: 下一时刻观测batch
            encoder_optimizer: 编码器优化器
            max_grad_norm: 最大梯度范数
            
        Returns:
            losses: 损失字典
        """
        # 1. 编码
        # 1.1 高程图VAE编码
        elevation_obs = obs_batch["height_scan_policy"]  # [B, T, H, W] 或 [B, T*H*W]
        # 整理高程图格式
        elevation_obs = elevation_obs.squeeze(1) if elevation_obs.dim() > 4 else elevation_obs
        elevation_features = self.elevation_vae_encoder(elevation_obs)
        
        mu_e = self.elevation_mu(elevation_features)
        logvar_e = self.elevation_logvar(elevation_features)
        logvar_e = torch.clamp(logvar_e, min=-10.0, max=10.0)  # 防止数值不稳定
        z_e = self.reparameterize(mu_e, logvar_e)
        
        # 1.2 本体历史编码（使用归一化后的policy观测 + mu_e）
        proprio_history = obs_batch["policy"]  # [B, T*obs_dim]（例如G1中480维）
        proprio_history_normalized = self.actor_obs_normalizer(proprio_history)
        proprio_features = self.proprio_encoder(proprio_history_normalized)
        
        proprio_mu_e_features = torch.cat([proprio_features, mu_e.detach()], dim=-1)
        v_hat = self.proprio_vel_head(proprio_mu_e_features)
        z_p = self.proprio_latent_head(proprio_mu_e_features)
        
        # 2. 解码
        # 2.1 高程图解码器：z_e -> 重建critic无噪声高程图
        elevation_recon = self.elevation_decoder(z_e)
        
        # 2.2 本体解码器：[z_e + z_p + v̂] -> 重建下一帧本体观测
        # z_e.detach(): 高程图编码已有自己的重建任务(elevation_recon_loss)
        # v_hat.detach(): 速度估计只受vel_loss监督，不受obs_recon_loss影响
        # z_p不detach: 让obs_recon_loss训练本体隐变量编码器
        decoder_input = torch.cat([z_e.detach(), z_p, v_hat.detach()], dim=-1)
        obs_recon = self.obs_decoder(decoder_input)
        
        # 3. 计算损失
        
        # 3.1 高程图重建损失（重建critic的无噪声高程图）
        # critic的高程图是无噪声的，我们要重建它
        elevation_target_critic = obs_batch["height_scan_critic"]  # critic的无噪声高程图
        
        # 确保形状一致
        if elevation_target_critic.dim() == 2:
            # 如果是展平的，确保尺寸匹配
            elevation_target_critic = elevation_target_critic[:, :self.elevation_decode_size]
        else:
            # 如果是4D，展平
            batch_size = elevation_target_critic.shape[0]
            elevation_target_critic = elevation_target_critic.view(batch_size, -1)[:, :self.elevation_decode_size]
        
        elevation_recon_loss = F.mse_loss(elevation_recon, elevation_target_critic)
        
        # 3.2 本体观测重建损失（重建下一帧本体观测）
        critic_obs_next = self.get_critic_obs(next_obs_batch)
        critic_obs_next_normalized = self.critic_obs_normalizer(critic_obs_next)
        obs_target = critic_obs_next_normalized[:, :self.num_decode]
        obs_recon_loss = F.mse_loss(obs_recon, obs_target)
        
        # 3.3 KL散度（VAE的正则化项）- 使用per-dimension平均而不是sum
        # 这样KL损失的scale与latent_dim无关
        kl_loss = -0.5 * torch.mean(1 + logvar_e - mu_e.pow(2) - logvar_e.exp())
        
        # 3.4 速度估计损失
        vel_target = critic_obs_next_normalized[:, -3:]  
        vel_loss = F.mse_loss(v_hat, vel_target)
        
        # 总损失
        total_loss = elevation_recon_loss + obs_recon_loss + self.kl_weight * kl_loss + vel_loss
        
        # 4. 反向传播和优化
        encoder_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)
        
        encoder_optimizer.step()
        
        # 5. 返回损失信息
        losses = {
            "elevation_recon_loss": elevation_recon_loss.item(),
            "obs_recon_loss": obs_recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "vel_loss": vel_loss.item(),
            "encoder_total_loss": total_loss.item(),
        }
        
        return losses
    
    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典
        """
        # Actor-Critic优化器
        std_params = [self.std] if self.noise_std_type == "scalar" else [self.log_std]
        actor_critic_params = list(self.actor.parameters()) + \
                             list(self.critic.parameters()) + \
                             list(self.critic_mlp_feature_extractor.parameters()) + \
                             list(self.critic_elevation_encoder.parameters()) + \
                             std_params
        
        actor_critic_optimizer = torch.optim.Adam(actor_critic_params, lr=learning_rate)
        
        # 编码器-解码器优化器
        encoder_params = list(self.elevation_vae_encoder.parameters()) + \
                        list(self.elevation_mu.parameters()) + \
                        list(self.elevation_logvar.parameters()) + \
                        list(self.proprio_encoder.parameters()) + \
                        list(self.proprio_vel_head.parameters()) + \
                        list(self.proprio_latent_head.parameters()) + \
                        list(self.elevation_decoder.parameters()) + \
                        list(self.obs_decoder.parameters())
        
        encoder_optimizer = torch.optim.Adam(encoder_params, lr=learning_rate)
        
        return {
            "optimizer": actor_critic_optimizer,
            "encoder_optimizer": encoder_optimizer
        }
    
    def reset(self, dones: torch.Tensor | None = None):
        """重置（如果需要）"""
        pass
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        # 训练时启用归一化层的统计更新
        self.actor_obs_normalizer.train(mode)
        self.critic_obs_normalizer.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        return self.train(False)

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode12P2_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将ElevationNet Mode12P2策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"ElevationNet_mode12P2_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建ElevationNet Mode12P2专用的导出器
        exporter = _ElevationNetMode12P2OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _ElevationNetMode12P2OnnxPolicyExporter(torch.nn.Module):
    """ElevationNet Mode12P2策略的ONNX导出器
    
    Mode12P2的推理流程:
    1. 从完整输入中分离：单帧本体观测 + 完整5帧本体历史 + 高程图历史
    2. 高程图编码：高程图 -> elevation_vae_encoder -> elevation_mu -> z_e
    3. 本体编码：5帧历史 -> proprio_encoder -> v_hat, z_p
    4. Actor推理：[单帧本体 + v_hat + z_e + z_p] -> actor -> actions
    """

    def __init__(self, policy: ActorCriticElevationNetMode12P2, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # 复制策略所需的模块
        # 高程图VAE编码器和mu head
        if hasattr(policy, "elevation_vae_encoder"):
            self.elevation_vae_encoder = copy.deepcopy(policy.elevation_vae_encoder)
        if hasattr(policy, "elevation_mu"):
            self.elevation_mu = copy.deepcopy(policy.elevation_mu)
        
        # 本体编码器和相关head
        if hasattr(policy, "proprio_encoder"):
            self.proprio_encoder = copy.deepcopy(policy.proprio_encoder)
        if hasattr(policy, "proprio_vel_head"):
            self.proprio_vel_head = copy.deepcopy(policy.proprio_vel_head)
        if hasattr(policy, "proprio_latent_head"):
            self.proprio_latent_head = copy.deepcopy(policy.proprio_latent_head)
        
        # Actor网络
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        # 保存维度信息
        self.elevation_sampled_frames = policy.elevation_sampled_frames
        self.vision_spatial_size = policy.vision_spatial_size
        self.num_proprio_one_frame = policy.num_proprio_one_frame
        self.num_actor_obs = policy.num_actor_obs  # 5帧历史维度
        self.latent_dim = policy.latent_dim
        self.vel_dim = policy.vel_dim
        self.proprio_latent_dim = policy.proprio_latent_dim
        
        # 计算各部分维度
        height, width = self.vision_spatial_size
        self.elevation_dim = self.elevation_sampled_frames * height * width

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_input):
        """前向传播（单输入版本）
        
        Args:
            obs_input: 合并的观测数据，形状为 [batch_size, total_obs_dim]
                       组成：[完整5帧历史 | 高程图]
                       - 完整5帧历史：num_actor_obs 维（例如G1中480维）
                       - 高程图：elevation_dim 维（例如5×25×17=2125维）
        
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_input.shape[0]
        
        # 切片分离各部分数据
        offset = 0
        # 1. 完整5帧历史（用于本体编码器和actor）
        proprio_history = obs_input[:, offset:offset + self.num_actor_obs]
        offset += self.num_actor_obs
        
        # 2. 高程图数据
        elevation_data_flat = obs_input[:, offset:]
        
        # 将高程图数据reshape为 [B, sampled_frames, height, width]
        height, width = self.vision_spatial_size
        elevation_data = elevation_data_flat.reshape(
            batch_size, self.elevation_sampled_frames, height, width
        )
        
        # 应用归一化器到完整5帧历史，然后提取单帧
        proprio_history_normalized = self.normalizer(proprio_history)
        current_proprio_obs = proprio_history_normalized[:, 0:self.num_proprio_one_frame]
        
        # 高程图编码：elevation -> z_e
        elevation_features = self.elevation_vae_encoder(elevation_data)
        z_e = self.elevation_mu(elevation_features)  # 推理时使用均值
        
        # 本体编码：5帧历史（归一化后） + z_e -> v_hat, z_p
        # 注意：训练时使用mu_e.detach()作为输入，推理时z_e本身就是detached
        proprio_features = self.proprio_encoder(proprio_history_normalized)
        proprio_mu_e_features = torch.cat([proprio_features, z_e], dim=-1)
        v_hat = self.proprio_vel_head(proprio_mu_e_features)
        z_p = self.proprio_latent_head(proprio_mu_e_features)
        
        # 融合特征
        fused_features = torch.cat([current_proprio_obs, v_hat, z_e, z_p], dim=-1)
        
        # 输出动作
        actions_mean = self.actor(fused_features)
        return actions_mean

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        
        # 计算总输入维度（不再包含单独的单帧本体）
        total_obs_dim = self.num_actor_obs + self.elevation_dim
        
        # 创建单个合并的输入示例
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (单输入模式 - Mode12P2):")
        print(f"{'='*80}")
        print(f"  5帧历史维度:      {self.num_actor_obs}")
        print(f"  高程图维度:       {self.elevation_dim} ({self.elevation_sampled_frames}×{self.vision_spatial_size[0]}×{self.vision_spatial_size[1]})")
        print(f"  总输入维度:       {total_obs_dim}")
        print(f"  ")
        print(f"  输入切片方式:")
        print(f"    [:, 0:{self.num_actor_obs}] = 5帧历史")
        print(f"    [:, {self.num_actor_obs}:] = 高程图")
        print(f"  ")
        print(f"  处理流程:")
        print(f"    1. 对5帧历史应用归一化 -> 提取单帧 ({self.num_proprio_one_frame}维)")
        print(f"    2. 高程图编码 -> z_e ({self.latent_dim}维)")
        print(f"    3. 5帧历史编码 -> v_hat ({self.vel_dim}维) + z_p ({self.proprio_latent_dim}维)")
        print(f"    4. Actor: [单帧 + v_hat + z_e + z_p] -> actions")
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

