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


class ActorCriticDWAQ(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        encoder_hidden_dims: tuple[int] | list[int] = [256, 256],
        decoder_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        num_decode: int = 30,
        num_latent: int = 19,
        num_history_len: int = 5,
        VAE_beta: float = 1.0,
        use_adaboot: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()

        # 记录beta值
        self.beta = VAE_beta

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        # 记录历史信息长度
        self.num_history_len = num_history_len
        # 单帧obs长度
        self.obs_one_frame_len: int = int(num_actor_obs / num_history_len)
        # 记录decoder输出的维度
        self.num_decoder = num_decode
        # 是否使用adaboot
        self.use_adaboot = use_adaboot

        # Actor
        if self.state_dependent_std:
            self.actor = MLP(self.obs_one_frame_len + num_latent, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(self.obs_one_frame_len + num_latent, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # 用于归一化历史本体信息
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Encoder
        self.encoder_backbone = MLP(num_actor_obs, encoder_hidden_dims[-1], encoder_hidden_dims[:-1], activation, last_activation="elu")
        self.encoder_latent_mean = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
        self.encoder_latent_logvar = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的logvar
        self.encoder_vel_mean = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值
        self.encoder_vel_logvar = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的logvar
        print(f"Encoder backbone MLP: {self.encoder_backbone}")
        print(f"Encoder latent mean: {self.encoder_latent_mean}")
        print(f"Encoder latent logvar: {self.encoder_latent_logvar}")
        print(f"Encoder velocity mean: {self.encoder_vel_mean}")
        print(f"Encoder velocity logvar: {self.encoder_vel_logvar}")

        # Decoder
        self.decoder = MLP(num_latent, num_decode, decoder_hidden_dims, activation)
        print(f"Decoder MLP: {self.decoder}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
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

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def reparameterise(self,mean,logvar):
        """重参数化

        Args:
            mean (_type_): 均值
            logvar (_type_): 对数方差

        Returns:
            _type_: 隐向量
        """
        std = torch.exp(logvar*0.5) # 得到标准差
        code_temp = torch.randn_like(std)
        code = mean + std * code_temp
        return code
    
    def encoder_forward(self,obs_history):
        """CENet 前向推理

        Args:
            obs_history (_type_): 历史观测值

        Returns:
            _type_: _description_
        """
        # 编码器网络前向推理
        x = self.encoder_backbone(obs_history)
        latent_mean = self.encoder_latent_mean(x) # 得到隐向量的均值
        latent_logvar = self.encoder_latent_logvar(x) # 得到隐向量的对数方差
        vel_mean = self.encoder_vel_mean(x) # 得到速度的均值
        vel_logvar = self.encoder_vel_logvar(x) # 得到速度的对数方差
        # 对数方差限制在一定范围内，避免过大
        latent_logvar = torch.clip(latent_logvar,min=-10,max=10)
        vel_logvar = torch.clip(vel_logvar,min=-10,max=10)
        # 采样隐向量和速度
        latent_sample = self.reparameterise(latent_mean,latent_logvar)
        vel_sample = self.reparameterise(vel_mean,vel_logvar)
        # 将速度和隐向量拼接起来
        code = torch.cat((vel_sample,latent_sample),dim=-1)
        # 解码得到下一时刻观测值
        decode = self.decoder(code)

        return code,vel_sample,latent_sample,decode,vel_mean,vel_logvar,latent_mean,latent_logvar

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        code,vel_sample,latent_sample,decode,vel_mean,vel_logvar,latent_mean,latent_logvar = self.encoder_forward(actor_obs)
        now_obs = actor_obs[:, 0:self.obs_one_frame_len]  # 取当前观测值部分
        
        # 根据条件决定是否使用adaboot
        reward = kwargs.get("rewards", None)
        if self.use_adaboot and reward is not None:
            # 计算adaboot概率
            CV_R = torch.std(reward) / (torch.mean(reward) + 1e-8)
            p_boot = 1 - torch.tanh(CV_R)
            # 获取真实线速度（使用原始obs）
            critic_obs = self.get_critic_obs(obs)
            critic_obs = self.critic_obs_normalizer(critic_obs)
            real_lin_vel = critic_obs[:, 0:3]  # 取前3维作为真实线速度
            # 按概率选择使用估计速度还是真实速度
            use_estimated = torch.rand(1, device=vel_sample.device).item() < p_boot
            selected_vel = vel_sample if use_estimated else real_lin_vel
            # observation = torch.cat((selected_vel.detach(), latent_sample.detach(), now_obs), dim=-1)
            observation = torch.cat((real_lin_vel.detach(), latent_sample.detach(), now_obs), dim=-1)
        else:
            # 不使用adaboot或在更新阶段，直接使用code
            observation = torch.cat((code.detach(), now_obs), dim=-1)
        
        self._update_distribution(observation)
        # 记录速度估计值
        self.extra_info["est_vel"] = vel_mean 
        self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decoder] + 1e-2) + self.actor_obs_normalizer.mean[:self.num_decoder]  # 返回反归一化后的预测观测值
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        code,vel_sample,latent_sample,decode,vel_mean,vel_logvar,latent_mean,latent_logvar = self.encoder_forward(obs)
        now_obs = obs[:, 0:self.obs_one_frame_len]  # 取当前观测值部分
        observation = torch.cat((vel_mean.detach(), latent_mean.detach(), now_obs),dim=-1)
        # 记录速度估计值
        self.extra_info["est_vel"] = vel_mean 
        self.extra_info["obs_predict"] = decode * (self.actor_obs_normalizer.std[:self.num_decoder] + 1e-2) + self.actor_obs_normalizer.mean[:self.num_decoder]  # 返回反归一化后的预测观测值
        if self.state_dependent_std:
            return self.actor(observation)[..., 0, :], self.extra_info
        else:
            return self.actor(observation), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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
