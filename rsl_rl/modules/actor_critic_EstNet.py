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
        num_latent = 27,
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
        super().__init__(        
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            **kwargs
        )

        # 获取一帧obs的长度
        self.one_obs_len: int = int(num_actor_obs / num_history_len)

        activation = resolve_nn_activation(activation)

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
        print(f"Encoder MLP: {self.encoder}")


    def estnet_forward(self,obs_history):
        """EstNet 前向推理
        Args:
            obs_history (_type_): 历史观测值

        """
        x = self.encoder(obs_history)
        latent = self.encode_latent(x) # 得到隐向量
        est_vel = self.encode_vel(x) # 得到速度估计值

        code = torch.cat((est_vel,latent),dim=-1)
        return code,est_vel

    def act(self, observations, **kwargs):
        """训练时使用的前向推理函数,action经过正态分布采样再输出

        Args:
            observations (_type_): 当前观测值
            obs_history (_type_): 观测值历史
        """
        code,_ = self.estnet_forward(observations)
        now_obs = observations[:, -self.one_obs_len:]
        observations = torch.cat((code.detach(),now_obs),dim=-1) # 隐向量放在当前观测值前面
        self.update_distribution(observations)
        return self.distribution.sample()


    def act_inference(self, observations):
        """部署时使用的前向推理函数

        Args:
            observations (_type_): _description_
            obs_history (_type_): _description_
        """

        code,_ = self.estnet_forward(observations)
        now_obs = observations[:, -self.one_obs_len:]
        observations = torch.cat((code.detach(),now_obs),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

