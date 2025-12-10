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


class ActorCriticEstNet(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        encoder_hidden_dims: tuple[int] | list[int] = [256, 256],
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        num_history_len: int = 5,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic_EstNet.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()

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

        # Actor
        if self.state_dependent_std:
            self.actor = MLP(self.obs_one_frame_len + 3, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(self.obs_one_frame_len + 3, num_actions, actor_hidden_dims, activation)
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
        self.encoder = MLP(num_actor_obs, 3, encoder_hidden_dims, activation)
        print(f"Encoder MLP: {self.encoder}")

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

    def encoder_forward(self,obs_history):
        """EstNet 前向推理
        Args:
            obs_history (_type_): 历史观测值

        """
        x = self.encoder(obs_history)
        return x

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        est_vel = self.encoder_forward(obs)
        now_obs = obs[:, 0:self.obs_one_frame_len]  # 取当前观测值部分
        observation = torch.cat((est_vel.detach(),now_obs),dim=-1)
        self._update_distribution(observation)
        # 记录速度估计值
        self.extra_info["est_vel"] = est_vel
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        est_vel = self.encoder_forward(obs)
        now_obs = obs[:, 0:self.obs_one_frame_len]  # 取当前观测值部分
        observation = torch.cat((est_vel.detach(),now_obs),dim=-1)
        # 记录速度估计值
        self.extra_info["est_vel"] = est_vel
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

    def update_encoder(
        self, 
        obs_batch: TensorDict,
        next_observations_batch: TensorDict,
        encoder_optimizer: torch.optim.Optimizer, 
        max_grad_norm: float
    ) -> dict[str, float]:
        """计算编码器损失并更新参数
        
        Args:
            obs_batch: 当前观测批次数据
            next_observations_batch: 下一时刻观测批次数据（EstNet不使用，但保持接口统一）
            encoder_optimizer: 编码器优化器
            max_grad_norm: 梯度裁剪的最大范数
            
        Returns:
            损失字典，包含各项损失值
        """
        # 获取并归一化policy观测
        policy_obs = self.get_actor_obs(obs_batch)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        
        # 前向传播得到速度估计
        vel_est = self.encoder_forward(policy_obs)
        
        # 获取并归一化critic观测，提取真实速度
        critic_obs = self.get_critic_obs(obs_batch)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        vel_target = critic_obs[:, 0:3]  # 真实速度作为目标
        vel_target.requires_grad = False
        
        # 计算MSE损失（放大1000倍以获得更明显的梯度）
        vel_MSE = nn.MSELoss()(vel_est, vel_target)
        
        # 反向传播
        encoder_optimizer.zero_grad()
        vel_MSE.backward(retain_graph=True)
        
        # 梯度裁剪
        encoder_params = [p for group in encoder_optimizer.param_groups for p in group['params']]
        nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)
        
        # 更新参数
        encoder_optimizer.step()
        
        # 返回统一格式的损失字典
        return {
            "vel_loss": vel_MSE.item(),
            "total_loss": vel_MSE.item(),
        }