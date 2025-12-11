# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticElevationNet(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # 网络架构模式选择
        network_mode: str = "mode1",  # "mode1", "mode2", "mode3"
        # 高程图MLP编码器配置(Mode2/3使用)
        vision_feature_dim: int = 64,
        vision_num_frames: int = 5,
        vision_spatial_size: tuple[int, int] = (11, 11),
        # 本体MLP编码器配置(Mode2/3使用)
        proprio_feature_dim: int = 128,
        # MLP融合网络配置(Mode2/3使用)
        fusion_mlp_hidden_dims: list[int] | None = None,
        # Mode3专用参数(估计器模式)
        encoder_hidden_dims: tuple[int] | list[int] = [1024, 512, 256],
        decoder_hidden_dims: tuple[int] | list[int] = [256, 512, 1024],
        num_latent: int = 19,  # 隐向量长度，包含3维线速度
        num_decode: int = 30,  # 解码器输出维度
        VAE_beta: float = 1.0,  # VAE的beta参数
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        # 配置类
        self.cfg = kwargs

        # 传递回Env的额外信息
        self.extra_info = dict()

        # 保存配置参数
        self.obs_groups = obs_groups
        self.vision_spatial_size = vision_spatial_size
        self.state_dependent_std = state_dependent_std
        self.network_mode = network_mode
        self.beta = VAE_beta
        
        # 计算本体观测和critic观测维度
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"] if g != "height_scan_history")
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"] if g != "height_scan_history")
        
        # 计算高程图展平后的维度
        height, width = vision_spatial_size
        height_map_dim = height * width

        ########################################## Actor ##############################################
        print("\n" + "=" * 80)
        print(f"🌟 网络架构模式: {network_mode}")
        print("=" * 80)
        
        if network_mode == "mode1":
            # Mode1: 本体观测和高程图拼接后进入一个MLP直接输出action
            print("✓ Mode1: 拼接 -> MLP -> Action")
            input_dim = num_actor_obs + height_map_dim
            self.direct_actor = MLP(input_dim, num_actions, actor_hidden_dims, activation)
            print(f"  Direct Actor: {input_dim} -> {num_actions}")
            
        elif network_mode == "mode2":
            # Mode2: 本体观测和高程图分别进MLP提取特征，再拼接+MLP融合后输出action
            print("✓ Mode2: 本体MLP -> 特征 + 高程图MLP -> 特征 + 拼接MLP -> Action")
            self.proprio_encoder = self._create_proprio_network(
                num_actor_obs, proprio_feature_dim, actor_hidden_dims, activation
            )
            self.elevation_net = self._create_perception_network(
                vision_feature_dim, vision_num_frames, vision_spatial_size
            )
            if fusion_mlp_hidden_dims is None:
                fusion_mlp_hidden_dims = [256, 128]
            self.fusion_actor = self._create_fusion_network(
                proprio_feature_dim, vision_feature_dim, num_actions, fusion_mlp_hidden_dims
            )
            height_map_input_dim = vision_num_frames * height_map_dim
            print(f"  Proprio MLP: {num_actor_obs} -> {proprio_feature_dim}")
            print(f"  Height Map MLP: {height_map_input_dim} ({vision_num_frames}×{height_map_dim}) -> {vision_feature_dim}")
            print(f"  Fusion MLP: {proprio_feature_dim + vision_feature_dim} -> {num_actions}")
            
        elif network_mode == "mode3":
            # Mode3: 类似Mode2，但输出隐向量(包括速度估计v和纯隐向量z)，类似DWAQ
            print("✓ Mode3: 本体MLP + 高程图MLP -> 拼接MLP -> 隐向量(v+z) -> Encoder/Decoder")
            self.proprio_encoder = self._create_proprio_network(
                num_actor_obs, proprio_feature_dim, actor_hidden_dims, activation
            )
            self.elevation_net = self._create_perception_network(
                vision_feature_dim, vision_num_frames, vision_spatial_size
            )
            
            # 融合网络输出隐向量特征
            fusion_output_dim = encoder_hidden_dims[-1]
            if fusion_mlp_hidden_dims is None:
                fusion_mlp_hidden_dims = [256, 128]
            self.fusion_encoder = self._create_fusion_network(
                proprio_feature_dim, vision_feature_dim, fusion_output_dim, fusion_mlp_hidden_dims
            )
            
            # Encoder分支：输出均值和方差
            self.encoder_latent_mean = nn.Linear(fusion_output_dim, num_latent - 3)
            self.encoder_latent_logvar = nn.Linear(fusion_output_dim, num_latent - 3)
            self.encoder_vel_mean = nn.Linear(fusion_output_dim, 3)
            self.encoder_vel_logvar = nn.Linear(fusion_output_dim, 3)
            
            # Decoder：从隐向量重建观测
            self.decoder = MLP(num_latent, num_decode, decoder_hidden_dims, activation)
            
            # Actor：从隐向量+当前本体观测输出动作
            self.num_decode = num_decode
            self.actor = MLP(num_latent + num_actor_obs, num_actions, actor_hidden_dims, activation)
            
            height_map_input_dim = vision_num_frames * height_map_dim
            print(f"  Proprio MLP: {num_actor_obs} -> {proprio_feature_dim}")
            print(f"  Height Map MLP: {height_map_input_dim} ({vision_num_frames}×{height_map_dim}) -> {vision_feature_dim}")
            print(f"  Fusion MLP: {proprio_feature_dim + vision_feature_dim} -> {fusion_output_dim}")
            print(f"  Latent: {num_latent} (vel: 3, latent: {num_latent-3})")
            print(f"  Decoder: {num_latent} -> {num_decode}")
            print(f"  Actor: {num_latent + num_actor_obs} -> {num_actions}")
        else:
            raise ValueError(f"Unknown network mode: {network_mode}. Should be 'mode1', 'mode2', or 'mode3'")
        
            print("=" * 80 + "\n")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()


        ########################################## Critic ##############################################
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        # Transformer Fusion架构不支持state_dependent_std(已在前面验证)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _create_proprio_network(
        self, 
        num_actor_obs: int, 
        proprio_feature_dim: int, 
        actor_hidden_dims: list[int], 
        activation: str
    ) -> nn.Module:
        """创建本体信息编码网络
        
        Mode2: 使用简单MLP提取本体特征
        Mode3: 同Mode2
        
        Args:
            num_actor_obs: 本体观测维度
            proprio_feature_dim: 输出特征维度
            actor_hidden_dims: 隐藏层维度列表
            activation: 激活函数名称
            
        Returns:
            本体编码器网络
        """
        return MLP(num_actor_obs, proprio_feature_dim, actor_hidden_dims, activation)
    
    def _create_perception_network(
        self,
        vision_feature_dim: int,
        vision_num_frames: int,
        vision_spatial_size: tuple[int, int]
    ) -> nn.Module:
        """创建感知网络(处理高程图序列)
        
        Mode2/3: 使用MLP处理展平的高程图序列
        
        Args:
            vision_feature_dim: 输出特征维度
            vision_num_frames: 帧数
            vision_spatial_size: 空间尺寸 (height, width)
            
        Returns:
            感知编码器网络
        """
        # 高程图展平后的维度 = frames * height * width
        height, width = vision_spatial_size
        input_dim = vision_num_frames * height * width
        
        # 使用MLP处理展平的高程图序列
        # 隐藏层维度根据输入输出自适应设置
        hidden_dims = [max(input_dim // 2, vision_feature_dim * 2), vision_feature_dim * 2]
        return MLP(input_dim, vision_feature_dim, hidden_dims, "elu")
    
    def _create_fusion_network(
        self,
        proprio_feature_dim: int,
        vision_feature_dim: int,
        num_actions: int,
        fusion_mlp_hidden_dims: list[int]
    ) -> nn.Module:
        """创建融合网络(拼接本体和感知特征并输出动作/隐向量)
        
        Mode2/3: 使用简单拼接+MLP融合两个特征
        
        Args:
            proprio_feature_dim: 本体特征维度
            vision_feature_dim: 视觉特征维度
            num_actions: 输出维度(Mode2为动作维度，Mode3为隐向量维度)
            fusion_mlp_hidden_dims: MLP的隐藏层维度
            
        Returns:
            融合网络
        """
        # 简单拼接两个特征后用MLP输出
        input_dim = proprio_feature_dim + vision_feature_dim
        return MLP(input_dim, num_actions, fusion_mlp_hidden_dims, "elu")

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

    def _extract_height_map(self, obs: TensorDict) -> torch.Tensor:
        """提取高程图的单帧展平数据(用于mode1)"""
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            keys = list(depth_obs.keys())
            depth_obs = depth_obs[keys[0]]
        # 取最新一帧: [batch, frames, height*width] -> [batch, height*width]
        return depth_obs[:, -1, :]
    
    def _extract_height_map_sequence(self, obs: TensorDict) -> torch.Tensor:
        """提取高程图序列并展平(用于mode2的MLP)"""
        depth_obs = obs["height_scan_history"]
        while isinstance(depth_obs, TensorDict):
            keys = list(depth_obs.keys())
            depth_obs = depth_obs[keys[0]]
        # 展平所有帧: [batch, frames, height*width] -> [batch, frames*height*width]
        batch_size = depth_obs.shape[0]
        return depth_obs.view(batch_size, -1)

    def reparameterise(self, mean, logvar):
        """重参数化技巧(用于mode3的VAE)"""
        std = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(std)
        code = mean + std * code_temp
        return code
    
    def encoder_forward(self, proprio_obs: torch.Tensor, obs: TensorDict):
        """Mode3编码器前向推理"""
        # 1. 提取本体特征
        proprio_features = self.proprio_encoder(proprio_obs)
        
        # 2. 提取并处理高程图序列(使用MLP)
        height_map_sequence = self._extract_height_map_sequence(obs)
        vision_features = self.elevation_net(height_map_sequence)
        
        # 3. 拼接两个特征后融合得到编码特征
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        x = self.fusion_encoder(fused_features)
        
        # 4. 分别输出速度和隐向量的均值和方差
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        vel_mean = self.encoder_vel_mean(x)
        vel_logvar = self.encoder_vel_logvar(x)
        
        # 限制方差范围
        latent_logvar = torch.clip(latent_logvar, min=-10, max=10)
        vel_logvar = torch.clip(vel_logvar, min=-10, max=10)
        
        # 采样
        latent_sample = self.reparameterise(latent_mean, latent_logvar)
        vel_sample = self.reparameterise(vel_mean, vel_logvar)
        
        # 拼接成完整隐向量
        code = torch.cat((vel_sample, latent_sample), dim=-1)
        
        # 解码
        decode = self.decoder(code)
        
        return code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar

    def _update_distribution(self, obs_input: torch.Tensor) -> None:
        """更新动作分布(统一接口，根据mode决定输入)"""
        mean = obs_input  # 对于mode1和mode2，obs_input已经是action mean
        
        # 计算标准差
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        if self.network_mode == "mode1":
            # Mode1: 拼接本体观测和高程图单帧，直接输入MLP
            height_map = self._extract_height_map(obs)
            actor_input = torch.cat([proprio_obs, height_map], dim=-1)
            mean = self.direct_actor(actor_input)
            self._update_distribution(mean)
            
        elif self.network_mode == "mode2":
            # Mode2: 本体和高程图序列分别提取特征，然后拼接后融合
            proprio_features = self.proprio_encoder(proprio_obs)
            height_map = self._extract_height_map(obs)
            vision_features = self.elevation_net(height_map)
            # 拼接两个特征
            fused_features = torch.cat([proprio_features, vision_features], dim=-1)
            mean = self.fusion_actor(fused_features)
            self._update_distribution(mean)
            
        elif self.network_mode == "mode3":
            # Mode3: 编码器输出隐向量，与当前观测拼接后输入actor
            code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
                self.encoder_forward(proprio_obs, obs)
            
            # 将code和当前本体观测拼接
            observation = torch.cat((code.detach(), proprio_obs), dim=-1)
            mean = self.actor(observation)
            self._update_distribution(mean)
            
            # 记录额外信息用于监控
            self.extra_info["est_vel"] = vel_mean
            self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                                       self.actor_obs_normalizer.mean[:self.num_decode]
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        if self.network_mode == "mode1":
            # Mode1: 拼接后直接推理
            height_map = self._extract_height_map(obs)
            actor_input = torch.cat([proprio_obs, height_map], dim=-1)
            mean = self.direct_actor(actor_input)
            
        elif self.network_mode == "mode2":
            # Mode2: 分别提取特征后拼接融合
            proprio_features = self.proprio_encoder(proprio_obs)
            height_map_sequence = self._extract_height_map_sequence(obs)
            vision_features = self.elevation_net(height_map_sequence)
            fused_features = torch.cat([proprio_features, vision_features], dim=-1)
            mean = self.fusion_actor(fused_features)
            
        elif self.network_mode == "mode3":
            # Mode3: 使用均值而非采样值
            code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
                self.encoder_forward(proprio_obs, obs)
        
            # 推理时使用均值
            observation = torch.cat((vel_mean.detach(), latent_mean.detach(), proprio_obs), dim=-1)
            mean = self.actor(observation)
            
            # 记录额外信息
            self.extra_info["est_vel"] = vel_mean
            self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decode] + 1e-2) + \
                                            self.actor_obs_normalizer.mean[:self.num_decode]
        
        return mean, self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取actor的本体观测(排除深度图)"""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            # 深度图单独处理，不加入本体观测
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic观测(排除深度图)"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            # 深度图不用于critic(可以根据需求修改)
            if obs_group != "height_scan_history":
                obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["critic"][0]].shape[0], 0)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
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
        """更新Mode3的编码器(仅在mode3下使用)

        Args:
            obs_batch: 当前观测批次数据
            next_observations_batch: 下一时刻观测批次数据
            encoder_optimizer: 编码器优化器
            max_grad_norm: 梯度裁剪的最大范数

        Returns:
            损失字典，包含各项损失值
        """
        if self.network_mode != "mode3":
            return {}

        # 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs = self.actor_obs_normalizer(policy_obs)

        # 前向传播得到编码器输出
        code, vel_sample, latent_sample, decode, vel_mean, vel_logvar, latent_mean, latent_logvar = \
            self.encoder_forward(policy_obs, obs_batch)

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
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
