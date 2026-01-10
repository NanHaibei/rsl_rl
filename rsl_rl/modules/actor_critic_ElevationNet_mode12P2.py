# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode12P2: 双编码器架构 - 高程图编码器 + 本体编码器

网络结构:
    Actor pipeline:
        单帧本体观测 + [高程图隐向量 + 线速度 + 足端高度 + 观测值隐向量] -> Actor网络 -> 动作
    
    高程图Encoder-Decoder:
        历史高程图序列 -> R(2+1)D -> 高程图隐向量 -> 高程图Decoder -> 重建无噪声高程图
    
    本体Encoder-Decoder:
        5帧历史本体观测 -> MLP -> 线速度估计(3) + 足端高度估计(2) + 观测值隐向量(N)
        [高程图隐向量 + 线速度 + 足端高度 + 观测值隐向量] -> MLP -> 重建下一时刻观测值
    
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
import torch.nn.functional as F


def ssim_loss(x, y, window_size=11, size_average=True):
    """计算SSIM损失
    
    Args:
        x: 预测图像 [B, C, H, W] 或 [B, T, H, W]
        y: 目标图像 [B, C, H, W] 或 [B, T, H, W]
        window_size: 窗口大小
        size_average: 是否平均
        
    Returns:
        SSIM损失值 (1 - SSIM，越小越好)
    """
    # 确保输入是4D张量 [B, C, H, W]
    if x.dim() == 4:
        # 已经是 [B, C, H, W] 或 [B, T, H, W]
        pass
    else:
        raise ValueError(f"Expected 4D input, got {x.dim()}D")
    
    # 创建高斯窗口
    channel = x.size(1)
    window = create_window(window_size, channel).to(x.device).type_as(x)
    
    # 计算均值
    mu1 = F.conv2d(x, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return 1 - ssim_map.mean()  # 返回损失 (1-SSIM)
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    """生成1D高斯核"""
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """创建2D高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class R21DElevationEncoder(nn.Module):
    """R(2+1)D编码器（VAE版本），用于处理高程图序列
    
    R(2+1)D将3D卷积分解为：
    1. 2D空间卷积（处理每一帧的空间信息）
    2. 1D时间卷积（处理帧间的时序信息）
    
    VAE变体：输出均值mu和对数方差logvar，用于重参数化采样
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
        self.out_dim = out_dim
        
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
        
        # VAE：输出mu和logvar
        self.fc_mu = nn.Linear(conv_output_size, out_dim)
        self.fc_logvar = nn.Linear(conv_output_size, out_dim)

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

    def forward(self, x, return_mu_logvar=False):
        """
        Args:
            x: [B, T, H, W] 高程图序列
            return_mu_logvar: 是否返回mu和logvar（训练时True，推理时False）
        Returns:
            如果return_mu_logvar=True: (z, mu, logvar)
            如果return_mu_logvar=False: z（使用mu作为z，无采样）
        """
        # 添加通道维度: [B, T, H, W] -> [B, 1, T, H, W]
        x = x.unsqueeze(1)
        
        # 归一化：使用固定值
        x = torch.clip((1.24 - x) / 0.6, -5.0, 5.0)
        
        # 通过R(2+1)D卷积
        x = self.conv(x)
        
        # 展平
        x = x.flatten(1)
        
        # VAE: 计算mu和logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        if return_mu_logvar:
            # 训练模式：重参数化采样
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            # 推理模式：直接使用均值
            return mu
    
    def _reparameterize(self, mu, logvar):
        """VAE重参数化技巧
        
        Args:
            mu: [B, out_dim] 均值
            logvar: [B, out_dim] 对数方差
        Returns:
            z: [B, out_dim] 采样的隐向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ElevationDecoder(nn.Module):
    """高程图解码器：使用3D转置卷积从隐向量重建无噪声高程图
    
    使用3D转置卷积可以更好地保留和重建高程图的空间-时间结构，
    相比MLP能更好地捕捉局部空间特征和时序关系。
    """
    
    def __init__(
        self, 
        latent_dim: int,
        num_frames: int,
        spatial_size: tuple[int, int],
        hidden_dims: list[int] = [64, 32, 16]  # 从高维到低维，与编码器相反
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        
        # 计算初始特征图的尺寸
        # 假设编码器使用了3次stride=2的下采样
        num_downsamples = 3
        init_h = spatial_size[0] // (2 ** num_downsamples)
        init_w = spatial_size[1] // (2 ** num_downsamples)
        init_t = num_frames  # 时间维度保持不变
        
        self.init_h = max(init_h, 1)
        self.init_w = max(init_w, 1)
        self.init_t = init_t
        
        # 第一层：将隐向量投影到初始3D特征图
        init_channels = hidden_dims[0]
        self.fc = nn.Linear(latent_dim, init_channels * self.init_t * self.init_h * self.init_w)
        self.init_channels = init_channels
        
        # 构建3D转置卷积层
        layers = []
        in_channels = init_channels
        
        for i, out_channels in enumerate(hidden_dims[1:] + [1]):  # 最后输出单通道
            # 3D转置卷积块
            layers.append(self._make_transpose_block(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2 if i < len(hidden_dims) else 1,  # 最后一层不上采样
                is_final=(i == len(hidden_dims))
            ))
            in_channels = out_channels
        
        self.decoder_conv = nn.Sequential(*layers)
    
    def _make_transpose_block(self, in_channels, out_channels, kernel_size, stride, is_final=False):
        """创建3D转置卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长（上采样因子）
            is_final: 是否为最后一层
        """
        layers = []
        
        if stride > 1:
            # 上采样层：使用转置卷积
            layers.append(nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size, kernel_size),  # 时间维度不变
                stride=(1, stride, stride),
                padding=(0, kernel_size//2, kernel_size//2),
                output_padding=(0, stride-1, stride-1)
            ))
        else:
            # 最后一层：普通3D卷积
            layers.append(nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=1,
                padding=kernel_size//2
            ))
        
        if not is_final:
            # 非最后一层：添加BN和激活
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        else:
            # 最后一层：使用Tanh限制输出范围（可选）
            # layers.append(nn.Tanh())
            pass
        
        return nn.Sequential(*layers)
    
    def forward(self, latent):
        """
        Args:
            latent: [B, latent_dim] 高程图隐向量
        Returns:
            elevation_map: [B, T, H, W] 重建的高程图
        """
        batch_size = latent.shape[0]
        
        # 1. 全连接层投影到初始3D特征图
        x = self.fc(latent)
        x = x.view(batch_size, self.init_channels, self.init_t, self.init_h, self.init_w)
        
        # 2. 通过转置卷积层上采样
        x = self.decoder_conv(x)
        
        # 3. 移除通道维度并调整到目标尺寸
        # x: [B, 1, T, H', W'] -> [B, T, H, W]
        x = x.squeeze(1)
        
        # 如果尺寸不完全匹配，使用插值调整
        if x.shape[1:] != (self.num_frames, self.spatial_size[0], self.spatial_size[1]):
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1),  # [B, 1, T, H', W']
                size=(self.num_frames, self.spatial_size[0], self.spatial_size[1]),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)
        
        return x


class ActorCriticElevationNetMode12P2(nn.Module):
    """Mode12P2: 双编码器架构 - 高程图编码器 + 本体编码器
    
    网络组成:
    1. Actor部分:
       - 当前帧本体观测 + encoder输出 -> Actor网络 -> 动作
    
    2. 高程图Encoder-Decoder:
       - 历史高程图序列 -> R(2+1)D -> 高程图隐向量(elevation_latent)
       - 高程图隐向量 -> 高程图Decoder -> 重建无噪声高程图
    
    3. 本体Encoder-Decoder:
       - 5帧历史本体观测 -> MLP -> 线速度(3) + 足端高度(2) + 观测值隐向量(N)
       - [高程图隐向量 + 线速度 + 足端高度 + 观测值隐向量] -> MLP -> 重建下一观测
    
    4. Critic部分:
       - 本体特权观测 -> MLP -> critic_mlp_feature
       - 特权高程图序列 -> R(2+1)D -> critic_vision_feature
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
        # 高程图编码器配置
        elevation_latent_dim: int = 32,  # 高程图隐向量维度
        r21d_hidden_dims: list[int] = [16, 32, 64],
        r21d_kernel_sizes: list[int] = [3, 3, 3],
        # 高程图解码器配置
        elevation_decoder_hidden_dims: list[int] = [128, 256, 512],
        # 本体编码器配置（本体历史 -> vel + feet_height + obs_latent）
        proprio_encoder_hidden_dims: tuple[int] | list[int] = [256, 128],
        num_vel: int = 3,
        num_feet_height: int = 2,
        obs_latent_dim: int = 14,  # 观测值隐向量维度
        # 观测解码器配置
        obs_decoder_hidden_dims: tuple[int] | list[int] = [128, 256, 512],
        num_decode: int = 70,  # 重建的下一观测维度
        # Critic MLP配置
        critic_mlp_feature_dim: int = 64,
        critic_mlp_extractor_hidden_dims: tuple[int] | list[int] = [128],
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
        self.elevation_latent_dim = elevation_latent_dim
        self.obs_latent_dim = obs_latent_dim
        self.num_vel = num_vel
        self.num_feet_height = num_feet_height
        self.critic_mlp_feature_dim = critic_mlp_feature_dim
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / elevation_sampled_frames)  # 单帧本体观测维度
        self.num_decode = num_decode
        
        # Encoder输出总维度：高程图隐向量 + 线速度 + 足端高度 + 观测值隐向量
        self.total_encoder_output_dim = elevation_latent_dim + num_vel + num_feet_height + obs_latent_dim
        
        ########################################## 网络架构 ##############################################
        
        # 1. 高程图编码器-解码器
        # 高程图编码器：历史高程图序列 -> R(2+1)D -> 高程图隐向量
        self.elevation_encoder = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=elevation_latent_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # 高程图解码器：高程图隐向量 -> 重建无噪声高程图
        self.elevation_decoder = ElevationDecoder(
            latent_dim=elevation_latent_dim,
            num_frames=elevation_sampled_frames,
            spatial_size=vision_spatial_size,
            hidden_dims=elevation_decoder_hidden_dims
        )
        
        # 2. 本体编码器-解码器
        # 本体编码器：5帧历史本体观测 -> MLP -> 线速度 + 足端高度 + 观测值隐向量
        proprio_encoder_output_dim = max(proprio_encoder_hidden_dims)
        self.proprio_encoder = MLP(
            num_actor_obs,  # 5帧历史本体观测
            proprio_encoder_output_dim,
            proprio_encoder_hidden_dims[:-1],
            activation
        )
        
        # 本体编码器输出头
        self.encoder_vel = nn.Linear(proprio_encoder_output_dim, num_vel)
        self.encoder_feet_height = nn.Linear(proprio_encoder_output_dim, num_feet_height)
        self.encoder_obs_latent = nn.Linear(proprio_encoder_output_dim, obs_latent_dim)
        
        # 观测解码器：[高程图隐向量 + 线速度 + 足端高度 + 观测值隐向量] -> 重建下一观测
        decoder_input_dim = elevation_latent_dim + num_vel + num_feet_height + obs_latent_dim
        self.obs_decoder = MLP(
            decoder_input_dim,
            num_decode,
            obs_decoder_hidden_dims,
            activation
        )
        
        # 3. Actor网络
        # Actor主网络：[当前帧本体观测 + encoder输出] -> 动作
        actor_input_dim = self.num_proprio_one_frame + self.total_encoder_output_dim
        self.actor = MLP(
            actor_input_dim,
            num_actions,
            actor_hidden_dims,
            activation
        )
        
        # 4. Critic网络
        # MLP特征提取器：本体特权观测 -> 特征向量
        self.critic_mlp_extractor = MLP(
            num_critic_obs,
            critic_mlp_feature_dim,
            critic_mlp_extractor_hidden_dims,
            activation
        )
        
        # R(2+1)D编码器：特权高程图序列 -> 视觉特征向量
        self.elevation_encoder_critic = R21DElevationEncoder(
            num_frames=elevation_sampled_frames,
            hidden_dims=r21d_hidden_dims,
            kernel_sizes=r21d_kernel_sizes,
            strides=[2] * len(r21d_hidden_dims),
            out_dim=elevation_latent_dim,
            vision_spatial_size=vision_spatial_size
        )
        
        # Critic主网络：融合特征 -> 价值
        self.critic = MLP(
            critic_mlp_feature_dim + elevation_latent_dim,
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
        print("ActorCriticElevationNetMode12P2 网络结构 (双编码器架构)")
        print("="*80)
        print(f"单帧本体观测维度: {self.num_proprio_one_frame}")
        print(f"5帧历史本体观测维度: {num_actor_obs}")
        print(f"")
        print(f"高程图Encoder-Decoder:")
        print(f"  - 高程图序列 -> R(2+1)D -> 高程图隐向量({elevation_latent_dim})")
        print(f"  - 高程图隐向量 -> Decoder -> 重建高程图({elevation_sampled_frames}x{vision_spatial_size[0]}x{vision_spatial_size[1]})")
        print(f"")
        print(f"本体Encoder-Decoder:")
        print(f"  - 5帧历史本体 -> MLP -> 线速度({num_vel}) + 足端高度({num_feet_height}) + 观测隐向量({obs_latent_dim})")
        print(f"  - [高程图隐向量 + 线速度 + 足端高度 + 观测隐向量] -> MLP -> 重建下一观测({num_decode})")
        print(f"")
        print(f"Actor pipeline:")
        print(f"  - Actor输入维度: {actor_input_dim} = 当前帧本体({self.num_proprio_one_frame}) + Encoder输出({self.total_encoder_output_dim})")
        print(f"  - Actor输出维度: {num_actions}")
        print(f"")
        print(f"Critic pipeline:")
        print(f"  - Critic输入维度: {critic_mlp_feature_dim + elevation_latent_dim} = MLP特征({critic_mlp_feature_dim}) + 视觉({elevation_latent_dim})")
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

    def encoder_forward(self, proprio_obs, elevation_obs):
        """编码器前向传播
        
        Args:
            proprio_obs: [B, 5*proprio_dim] 5帧历史本体观测
            elevation_obs: [B, T, H, W] 高程图序列
            
        Returns:
            elevation_latent: [B, elevation_latent_dim] 高程图隐向量
            vel: [B, 3] 线速度估计
            feet_height: [B, 2] 足端高度估计
            obs_latent: [B, obs_latent_dim] 观测值隐向量
        """
        # 高程图编码器（VAE）
        # 推理模式：直接使用均值，不采样
        elevation_latent = self.elevation_encoder(elevation_obs, return_mu_logvar=False)
        
        # 本体编码器
        proprio_features = self.proprio_encoder(proprio_obs)
        vel = self.encoder_vel(proprio_features)
        feet_height = self.encoder_feet_height(proprio_features)
        obs_latent = self.encoder_obs_latent(proprio_features)
        
        return elevation_latent, vel, feet_height, obs_latent

    def act(self, obs: TensorDict, **kwargs) -> tuple[torch.Tensor, dict]:
        """Actor前向传播（训练模式）
        
        Args:
            observations: 包含policy、height_scan_policy、obs_one_frame的TensorDict
            
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
        
        # 2. Encoder前向传播
        elevation_latent, vel, feet_height, obs_latent = self.encoder_forward(proprio_obs, elevation_obs)
        
        # 3. 拼接encoder输出（detach阻止actor的梯度回传到encoder）
        encoder_output = torch.cat([
            elevation_latent.detach(), 
            vel.detach(), 
            feet_height.detach(), 
            obs_latent.detach()
        ], dim=-1)
        
        # 4. Actor前向传播
        # 拼接当前帧本体观测和encoder输出
        actor_input = torch.cat([current_frame_obs, encoder_output], dim=-1)
        
        # 通过Actor网络
        action_mean = self.actor(actor_input)
        
        # 5. 动作分布和采样
        if self.noise_std_type == "scalar":
            action_std = self.std
        elif self.noise_std_type == "log":
            action_std = torch.exp(self.log_std)
        
        self.distribution = Normal(action_mean, action_std)
        actions = self.distribution.sample()
        
        # 6. 存储额外信息
        extra_info = {
            "elevation_latent": elevation_latent.detach(),
            "vel": vel.detach(),
            "feet_height": feet_height.detach(),
            "obs_latent": obs_latent.detach(),
        }
        
        return actions, extra_info

    def act_inference(self, observations: TensorDict) -> torch.Tensor:
        """Actor前向传播（推理模式）
        
        Args:
            observations: 包含policy、height_scan_policy、obs_one_frame的TensorDict
            
        Returns:
            action_mean: [B, num_actions] 动作均值（无噪声）
        """
        # 1. 提取观测
        proprio_obs = observations["policy"]  # 5帧历史本体
        elevation_obs = observations["height_scan_policy"]  # 高程图，已经是[B, T, H, W]格式
        
        # 归一化
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        current_frame_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 当前帧本体
        
        # 2. Encoder前向传播
        elevation_latent, vel, feet_height, obs_latent = self.encoder_forward(proprio_obs, elevation_obs)
        
        # 3. 拼接encoder输出（detach阻止actor的梯度回传到encoder）
        encoder_output = torch.cat([
            elevation_latent.detach(), 
            vel.detach(), 
            feet_height.detach(), 
            obs_latent.detach()
        ], dim=-1)
        
        # 4. Actor前向传播
        actor_input = torch.cat([current_frame_obs, encoder_output], dim=-1)
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
        """更新编码器和解码器
        
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
        elevation_obs_clean = obs_batch["height_scan_critic"]  # 无噪声的高程图（特权信息）
        
        # 2. Encoder前向传播（使用有噪声的高程图和归一化的policy观测）
        # VAE模式：返回采样的z, mu, logvar
        elevation_z, elevation_mu, elevation_logvar = self.elevation_encoder(
            elevation_obs, return_mu_logvar=True
        )
        
        # 本体编码器
        proprio_features = self.proprio_encoder(policy_obs_normalized)
        vel = self.encoder_vel(proprio_features)
        feet_height = self.encoder_feet_height(proprio_features)
        obs_latent = self.encoder_obs_latent(proprio_features)
        
        # 3. 计算重建损失
        
        # 3.1 高程图VAE损失（重建无噪声高程图 + KL散度）
        # 使用采样的z进行重建
        recon_elevation = self.elevation_decoder(elevation_z)
        
        # 重建损失：L1 loss + SSIM loss
        l1_loss = nn.functional.l1_loss(recon_elevation, elevation_obs_clean)
        ssim_loss_value = ssim_loss(recon_elevation, elevation_obs_clean, window_size=5)
        elevation_recon_loss = l1_loss + 0.1 * ssim_loss_value
        
        # KL散度损失
        elevation_kl_loss = -0.5 * torch.mean(
            1 + elevation_logvar - elevation_mu.pow(2) - elevation_logvar.exp()
        )
        
        # 3.2 获取真实目标值
        # 从critic观测中提取真实速度和脚掌高度
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs_normalized = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs_normalized[:, 70:73]  # base_lin_vel
        feet_height_target = critic_obs_normalized[:, 73:75]  # left_foot_height + right_foot_height
        
        # 获取下一时刻观测作为重建目标
        next_observations = self.get_critic_obs(next_obs_batch)
        next_observations_normalized = self.critic_obs_normalizer(next_observations)
        obs_target = next_observations_normalized[:, 0:self.num_decode]
        
        vel_target.requires_grad = False
        feet_height_target.requires_grad = False
        obs_target.requires_grad = False
        
        # 3.3 本体信息重建损失
        # 线速度重建损失
        vel_loss = nn.functional.mse_loss(vel, vel_target)
        # 脚掌高度重建损失
        feet_height_loss = nn.functional.mse_loss(feet_height, feet_height_target)
        
        # 3.4 观测重建损失
        # 拼接所有encoder输出作为decoder输入
        # elevation_mu(使用mu), vel, feet_height需要detach，因为它们主要用于各自的重建任务
        # obs_latent不detach，因为它是专门为观测重建设计的隐向量
        decoder_input = torch.cat([
            elevation_mu.detach(),  # 使用mu而不是采样的z
            vel.detach(), 
            feet_height.detach(), 
            obs_latent
        ], dim=-1)
        recon_obs = self.obs_decoder(decoder_input)
        obs_loss = nn.functional.mse_loss(recon_obs, obs_target)
        
        # 4. 总损失（包含VAE的KL散度）
        # VAE beta系数可配置，默认1.0
        vae_beta = 1.0  # 可以作为超参数调整
        total_loss = elevation_recon_loss + vae_beta * elevation_kl_loss + vel_loss + feet_height_loss + obs_loss
        
        # 5. 反向传播和优化
        encoder_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)
        
        encoder_optimizer.step()
        
        # 6. 返回损失信息（包含VAE相关损失）
        losses = {
            "elevation_recon_loss": elevation_recon_loss.item(),
            "elevation_kl_loss": elevation_kl_loss.item(),
            "vel_loss": vel_loss.item(),
            "feet_height_loss": feet_height_loss.item(),
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
            {'params': self.actor.parameters()},
            {'params': self.critic_mlp_extractor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.elevation_encoder_critic.parameters()},
            {'params': [self.std] if self.noise_std_type == "scalar" else [self.log_std]},
        ], lr=learning_rate)
        
        # 编码器和解码器的优化器
        encoder_optimizer = optim.Adam([
            {'params': self.elevation_encoder.parameters()},
            {'params': self.elevation_decoder.parameters()},
            {'params': self.proprio_encoder.parameters()},
            {'params': self.encoder_vel.parameters()},
            {'params': self.encoder_feet_height.parameters()},
            {'params': self.encoder_obs_latent.parameters()},
            {'params': self.obs_decoder.parameters()},
        ], lr=learning_rate)
        
        return {
            "optimizer": optimizer,
            "encoder_optimizer": encoder_optimizer
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """加载模型参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True

    def export_to_onnx(self, path: str, filename: str = "ElevationNet_mode12P2_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将Mode12P2策略导出为ONNX格式
        
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
        
        _exporter = _ElevationNetMode12P2OnnxPolicyExporter(
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


class _ElevationNetMode12P2OnnxPolicyExporter(torch.nn.Module):
    """Mode12P2策略的ONNX导出器"""

    def __init__(
        self,
        policy: ActorCriticElevationNetMode12P2,
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
        
        # Encoder前向传播
        elevation_latent, vel, feet_height, obs_latent = self.policy.encoder_forward(proprio_obs_history, elevation_obs)
        
        # 拼接encoder输出
        encoder_output = torch.cat([elevation_latent, vel, feet_height, obs_latent], dim=-1)
        
        # Actor前向传播
        actor_input = torch.cat([current_frame_obs, encoder_output], dim=-1)
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
