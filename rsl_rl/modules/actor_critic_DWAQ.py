# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic

class ActorCritic_DWAQ(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_latent = 32,
        encoder_hidden_dims=[256, 256],
        decoder_hidden_dims=[256, 256, 256],
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        num_history_len = 5,
        **kwargs,
    ):
        super()._init_nn()
        # 返回已经实例化的激活函数类
        activation = resolve_nn_activation(activation)

        # 获取一帧obs的长度
        self.one_obs_len: int = int(num_actor_obs / num_history_len)

        mlp_input_dim_a = self.one_obs_len + num_latent # actor网络输入维度是观测值和隐向量的拼接
        mlp_input_dim_c = num_critic_obs
        # Policy 构建actor网络
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
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

        # TODO:是否可以只用两个头而不是四个
        self.encode_mean_latent = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
        self.encode_logvar_latent = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的对数方差
        self.encode_mean_vel = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值
        self.encode_logvar_vel = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的对数方差

        # 构建decoder网络
        decoder_layers = []
        decoder_layers.append(nn.Linear(num_latent, decoder_hidden_dims[0]))
        decoder_layers.append(activation)
        for layer_index in range(len(decoder_hidden_dims)):
            if layer_index == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], num_actor_obs)) # 最后输出下一时刻的观测值
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_hidden_dims[layer_index + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)

        # Value function 构建critic网络
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 输出四个网络的结构
        print(f"Actor MLP: {self.actor}")
        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")
        print(f"Critic MLP: {self.critic}")

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

    def cenet_forward(self,obs_history):
        """CENet 前向推理

        Args:
            obs_history (_type_): 历史观测值

        Returns:
            _type_: _description_
        """
        x = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(x) # 得到隐向量的均值
        logvar_latent = self.encode_logvar_latent(x) # 得到隐向量的对数方差
        logvar_latent = torch.clip(logvar_latent,min=-10,max=10)
        mean_vel = self.encode_mean_vel(x) # 得到速度的均值
        logvar_vel = self.encode_logvar_vel(x) # 得到速度的对数方差
        logvar_vel = torch.clip(logvar_vel,min=-10,max=10)

        # code_latent = self.reparameterise(mean_latent,logvar_latent)
        # code_vel = self.reparameterise(mean_vel,logvar_vel)
        code_latent = mean_latent
        code_vel = mean_vel

        # code = torch.cat((code_vel,code_latent),dim=-1)
        code = code_vel
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    def act(self, obs_history, **kwargs):
        """训练时使用的前向推理函数,policy输出经过正态分布再

        Args:
            observations (_type_): 当前观测值
            obs_history (_type_): 观测值历史
        """
        code,_,_,_,_,_,_ = self.cenet_forward(obs_history)
        now_obs = obs_history[:, -self.one_obs_len:]
        critic_obs = kwargs.get("critic_obs", None)
        real_vel = critic_obs[:, 0:3]
        observations = torch.cat((code.detach(), now_obs),dim=-1) # 隐向量放在当前观测值前面
        # observations = torch.cat((real_vel.detach(),now_obs),dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()



    def act_inference(self, obs_history):
        """部署时使用的前向推理函数

        Args:
            observations (_type_): _description_
            obs_history (_type_): _description_
        """

        x = self.encoder(obs_history)
        mean_vel = self.encode_mean_vel(x)
        mean_latent = self.encode_mean_latent(x)
        code = torch.cat((mean_vel,mean_latent),dim=-1)
        now_obs = obs_history[:, -self.one_obs_len:]
        observations = torch.cat((code.detach(), now_obs),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean
