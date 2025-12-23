# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticElevationNetMode10: VAE编码器架构，使用2D CNN处理高程图序列

网络结构:
    本体 -> 本体编码器MLP -> 特征1
    高程图序列 -> 2D CNN编码器(历史帧作为通道) -> 特征2
    [特征1 + 特征2] -> 融合MLP -> 编码特征
    编码特征 -> Encoder -> 隐向量(v+z)
    隐向量 -> Decoder -> 重建观测
    [隐向量 + 本体] -> Actor MLP -> 动作
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

class ActorCriticElevationNetMode10(nn.Module):
    """Mode10: 使用2D CNN处理高程图序列（历史帧作为通道）+ VAE架构
    
    网络组成:
    1. 本体编码器MLP - 提取本体特征
    2. 高程图2D CNN编码器 - 将历史帧作为通道提取空间特征
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
        num_latent: int = 19,
        num_decode: int = 70,
        VAE_beta: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        # 配置
        self.cfg = kwargs
        self.extra_info = dict()
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size
        self.noise_std_type = noise_std_type
        self.history_frames = history_frames
        self.vision_feature_dim = vision_feature_dim
        self.beta = VAE_beta
        self.num_decode = num_decode
        self.num_encoder_output = num_encoder_output
        
        # 计算观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.num_proprio_one_frame = int(num_actor_obs / history_frames)
        
        ########################################## 网络架构 ##############################################
        # 1. Actor网络
        self.actor = MLP(self.num_proprio_one_frame + num_latent, num_actions, actor_hidden_dims, activation)
        
        # 2. Critic网络
        self.elevation_encoder_critic = Conv2DElevationEncoder(
            num_frames=history_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.critic = MLP(num_critic_obs + vision_feature_dim, 1, critic_hidden_dims, activation)

        # 3. encoder-特征提取
        self.elevation_feature = Conv2DElevationEncoder(
            num_frames=history_frames,
            hidden_dims=conv2d_hidden_dims,
            kernel_sizes=conv2d_kernel_sizes,
            strides=[2] * len(conv2d_hidden_dims),  # 默认使用stride=2
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        self.proprio_feature = MLP(num_actor_obs, proprio_feature_dim, proprio_encoder_hidden_dims, activation)
        
        # 4. encoder-VAE
        self.encoder = MLP(proprio_feature_dim + vision_feature_dim, num_encoder_output, encoder_hidden_dims, activation)
        self.encoder_latent_mean = nn.Linear(num_encoder_output, num_latent - 3)
        self.encoder_latent_logvar = nn.Linear(num_encoder_output, num_latent - 3)
        self.encoder_vel_mean = nn.Linear(num_encoder_output, 3)
        self.encoder_vel_logvar = nn.Linear(num_encoder_output, 3)

        self.decoder = MLP(num_latent, num_decode, decoder_hidden_dims, activation)

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
        print("ActorCriticElevationNetMode10 网络结构")
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

    def _extract_height_map_sequence(self, obs: TensorDict) -> torch.Tensor:
        """提取高程图序列用于2D CNN处理（历史帧作为通道）"""
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            keys = list(depth_obs.keys())
            depth_obs = depth_obs[keys[0]]
        
        # depth_obs 形状: [batch_size, history_frames, height, width]
        # 这个形状适合Conv2DEncoder处理（历史帧将作为通道）
        return depth_obs

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(std)
        code = mean + std * code_temp
        return code
    
    def encoder_forward(self, proprio_obs: torch.Tensor, height_maps_obs: TensorDict):
        """编码器前向传播"""
        # 1. 提取特征
        proprio_features = self.proprio_feature(proprio_obs)
        vision_features = self.elevation_feature(height_maps_obs)
        
        # 3. 融合特征并通过encoder MLP
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        x = self.encoder(fused_features)
        
        # 4. VAE编码: 输出均值和方差
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        vel_mean = self.encoder_vel_mean(x)
        vel_logvar = self.encoder_vel_logvar(x)
        
        # 限制方差范围
        latent_logvar = torch.clip(latent_logvar, min=-10, max=10)
        vel_logvar = torch.clip(vel_logvar, min=-10, max=10)
        
        # 5. 采样隐向量
        latent_sample = self.reparameterise(latent_mean, latent_logvar)
        vel_sample = self.reparameterise(vel_mean, vel_logvar)
        
        # 6. 拼接成完整隐向量
        code = torch.cat((vel_sample, latent_sample), dim=-1)
        
        # 7. 解码
        decode = self.decoder(code)
        
        return code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar

    def _update_distribution(self, mean: torch.Tensor) -> None:
        """更新动作分布"""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """训练时的动作采样"""
        # 1. 获取并归一化本体观测 获取高程图观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        height_maps_obs = obs["height_scan_policy"]
        
        # 2. 编码器前向传播得到隐向量
        code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
            self.encoder_forward(proprio_obs, height_maps_obs)
        
        # 3. 将隐向量与当前本体观测拼接
        now_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 取当前观测值部分
        observation = torch.cat((code.detach(), now_obs), dim=-1)
        
        # 4. Actor输出动作
        mean = self.actor(observation)
        self._update_distribution(mean)
        
        # 5. 记录额外信息用于监控
        self.extra_info["est_vel"] = vel_mean
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                                       self.actor_obs_normalizer.mean[:self.num_decode]
        else:
            self.extra_info["obs_predict"] = decode
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        """推理时的确定性动作"""
        # 1. 获取并归一化本体观测 获取高程图观测
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        height_maps_obs = obs["height_scan_policy"]
        
        # 2. 编码器前向传播得到隐向量
        code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
            self.encoder_forward(proprio_obs, height_maps_obs)
        
        # 3. 推理时使用均值而非采样值
        now_obs = proprio_obs[:, 0:self.num_proprio_one_frame]  # 取当前观测值部分
        observation = torch.cat((vel_mean.detach(), latent_mean.detach(), now_obs), dim=-1)
        
        # 4. Actor输出确定性动作
        mean = self.actor(observation)
        
        # 5. 记录额外信息
        self.extra_info["est_vel"] = vel_mean
        if self.actor_obs_normalization:
            self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                            self.actor_obs_normalizer.mean[:self.num_decode]
        else:
            self.extra_info["obs_predict"] = decode
        
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值"""
        # 1. 提取观测值
        height_maps_obs = obs["height_scan_history"]
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
        # 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        height_maps_obs = obs_batch["height_scan_policy"]

        # 前向传播得到编码器输出
        code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
            self.encoder_forward(policy_obs, height_maps_obs)

        # 获取并归一化critic观测，提取真实速度
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs[:, 0:3]  # 真实速度作为目标

        # 获取下一时刻观测，提取目标观测
        next_observations = self.get_actor_obs(next_observations_batch)
        next_observations = self.actor_obs_normalizer(next_observations)
        obs_target = next_observations[:, 0:self.num_decode]  # 取最新观测

        vel_target.requires_grad = False
        obs_target.requires_grad = False

        # 损失计算：速度重建损失 + obs重建损失 + KL散度损失
        vel_MSE = nn.MSELoss()(vel_sample, vel_target) * 100.0
        # 确保decode和obs_target维度匹配
        obs_MSE = nn.MSELoss()(decode, obs_target)
        dkl_loss = -0.5 * torch.mean(torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1))
        autoenc_loss = vel_MSE + obs_MSE + self.beta * dkl_loss

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
            "obs_loss": obs_MSE.item(),
            "dkl_loss": dkl_loss.item(),
            "total_loss": autoenc_loss.item(),
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
            {'params': self.encoder_latent_mean.parameters()},
            {'params': self.encoder_latent_logvar.parameters()},
            {'params': self.encoder_vel_mean.parameters()},
            {'params': self.encoder_vel_logvar.parameters()},
            {'params': self.decoder.parameters()},
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
        if hasattr(policy, "encoder_latent_mean"):
            self.encoder_latent_mean = copy.deepcopy(policy.encoder_latent_mean)
        if hasattr(policy, "encoder_latent_logvar"):
            self.encoder_latent_logvar = copy.deepcopy(policy.encoder_latent_logvar)
        if hasattr(policy, "encoder_vel_mean"):
            self.encoder_vel_mean = copy.deepcopy(policy.encoder_vel_mean)
        if hasattr(policy, "encoder_vel_logvar"):
            self.encoder_vel_logvar = copy.deepcopy(policy.encoder_vel_logvar)
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
        latent_mean = self.encoder_latent_mean(x)
        vel_mean = self.encoder_vel_mean(x)
        
        # 拼接隐向量
        code = torch.cat((vel_mean, latent_mean), dim=-1)
        
        # 与当前本体观测拼接
        now_obs = normalized_obs[:, 0:self.num_proprio_one_frame]
        observation = torch.cat((code.detach(), now_obs), dim=-1)
        
        # 输出动作
        actions_mean = self.actor(observation)
        return actions_mean, vel_mean

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
            output_names=["actions", "est_vel"],
            dynamic_axes={},
        )
