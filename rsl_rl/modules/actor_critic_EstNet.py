# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic import ActorCritic

class ActorCritic_EstNet(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_latent = 19,
        encoder_hidden_dims=[256, 256],
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        num_history_len = 5,
        **kwargs,
    ):
        # 初始化父类
        # super().__init__(        
        #     num_actor_obs,
        #     num_critic_obs,
        #     num_actions,
        #     actor_hidden_dims,
        #     critic_hidden_dims,
        #     activation,
        #     init_noise_std,
        #     noise_std_type,
        #     print_networks=print_networks,
        #     **kwargs
        # )
        if kwargs:
            print(
                "ActorCritic_EstNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # 初始化nn.Module
        self._init_nn()
        
        activation = resolve_nn_activation(activation)
        # 获取一帧obs的长度
        self.one_obs_len: int = int(num_actor_obs / num_history_len)

        # Policy 构建actor网络
        actor_layers = []
        actor_layers.append(nn.Linear(self.one_obs_len + num_latent, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function 构建critic网络
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 构建encoder网络
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_actor_obs, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for layer_index in range(len(encoder_hidden_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
            encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        self.encode_latent = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
        self.encode_vel = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值

        # 输出网络结构
        print(f"Actor MLP: {self.actor}")
        print(f"Encoder MLP: {self.encoder}")
        print(f"Encoder latent: {self.encode_latent}")
        print(f"Encoder velocity: {self.encode_vel}")

        # Action noise 设置正态分布的初始标准差
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)


    def encoder_forward(self,obs_history):
        """EstNet 前向推理
        Args:
            obs_history (_type_): 历史观测值

        """
        x = self.encoder(obs_history)
        latent = self.encode_latent(x) # 得到隐向量
        est_vel = self.encode_vel(x) # 得到速度估计值

        code = torch.cat((est_vel,latent),dim=-1)
        return code,est_vel

    def act(self, obs_history, **kwargs):
        """训练时使用的前向推理函数,action经过正态分布采样再输出

        Args:
            obs_history (_type_): 当前观测值
            obs_history (_type_): 观测值历史
        """
        code,_ = self.encoder_forward(obs_history)
        now_obs = obs_history[:, 0:self.one_obs_len]

        # TODO:对critic进行检查，real_vel不正确设置就报错
        # 实现adaboot
        # critic_obs = kwargs.get("critic_obs", None)
        # rewards = kwargs.get("rewards", None)
        # real_vel = critic_obs[:, 0:3]
        # if critic_obs is not None and rewards is not None:
        #     real_vel = critic_obs[:, 0:3]
        #     rewards = torch.clamp(rewards, 0, 99999) # 防止负的奖励进入std函数
        #     CV_R = torch.std(rewards) / torch.mean(rewards)
        #     p_boot = 1 - torch.tanh(CV_R)
        #     random_tensor = torch.rand(real_vel.shape, device=rewards.device)
        #     code[:, 0:3] = torch.where(random_tensor < p_boot.expand_as(random_tensor), real_vel, code[:, 0:3])
            # 完全使用真实值，用于debug
            # code[:, 0:3] = real_vel
        
        observations = torch.cat((code.detach(),now_obs),dim=-1) # 隐向量放在当前观测值前
        # observations = torch.cat((real_vel.detach(),now_obs),dim=-1) # 隐向量放在当前观测值前
        # observations = now_obs # 隐向量放在当前观测值前
        self.update_distribution(observations)
        return self.distribution.sample()


    def act_inference(self, obs_history):
        """部署时使用的前向推理函数

        Args:
            obs_history (_type_): _description_
            obs_history (_type_): _description_
        """

        code,_ = self.encoder_forward(obs_history)
        now_obs = obs_history[:, -self.one_obs_len:]
        observations = torch.cat((code.detach(),now_obs),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

