# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .actor_critic_EstNet import ActorCritic_EstNet, ActorCritic

class ActorCritic_DWAQ(ActorCritic_EstNet):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_latent = 19,
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
        # 初始化父类
        # super().__init__(        
        #     num_actor_obs,
        #     num_critic_obs,
        #     num_actions,
        #     num_latent,
        #     encoder_hidden_dims,
        #     actor_hidden_dims,
        #     critic_hidden_dims,
        #     activation,
        #     init_noise_std,
        #     noise_std_type,
        #     num_history_len,
        #     **kwargs
        # )

        if kwargs:
            print(
                "ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        # 初始化nn.Module
        self._init_nn()

        # 返回已经实例化的激活函数类
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

        #  构建encoder网络
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_actor_obs, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for layer_index in range(len(encoder_hidden_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
            encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_latent_mean = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
        self.encoder_latent_logvar = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
        self.encoder_vel_mean = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值
        self.encoder_vel_logvar = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值

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

        # 输出四个网络的结构
        print(f"Actor MLP: {self.actor}")
        print(f"Encoder MLP: {self.encoder}")
        print(f"Encoder latent mean: {self.encoder_latent_mean}")
        print(f"Encoder latent logvar: {self.encoder_latent_logvar}")
        print(f"Encoder velocity mean: {self.encoder_vel_mean}")
        print(f"Encoder velocity logvar: {self.encoder_vel_logvar}")
        print(f"Decoder MLP: {self.decoder}")

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

    def encoder_forward(self,obs_history):
        """CENet 前向推理

        Args:
            obs_history (_type_): 历史观测值

        Returns:
            _type_: _description_
        """
        # 编码器网络前向推理
        x = self.encoder(obs_history)
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

        return code,vel_sample,decode,vel_mean,vel_logvar,latent_mean,latent_logvar

    def act(self, obs_history, **kwargs):
        """训练时使用的前向推理函数,policy输出经过正态分布再

        Args:
            observations (_type_): 当前观测值
            obs_history (_type_): 观测值历史
        """
        code,_,_,_,_,_,_ = self.encoder_forward(obs_history)
        now_obs = obs_history[:, 0:self.one_obs_len]
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
        # critic_obs = kwargs.get("critic_obs", None)
        # real_vel = critic_obs[:, 0:3]
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
        mean_vel = self.encoder_vel_mean(x)
        mean_latent = self.encoder_latent_mean(x)
        code = torch.cat((mean_vel,mean_latent),dim=-1)
        now_obs = obs_history[:, -self.one_obs_len:]
        observations = torch.cat((code.detach(), now_obs),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

# class ActorCritic_DWAQ(ActorCritic):
#     is_recurrent = False

#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         num_latent = 19,
#         encoder_hidden_dims=[256, 256],
#         decoder_hidden_dims=[256, 256, 256],
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="elu",
#         init_noise_std=1.0,
#         noise_std_type: str = "scalar",
#         num_history_len = 5,
#         **kwargs,
#     ):
#         # 初始化父类
#         super().__init__(        
#             num_actor_obs,
#             num_critic_obs,
#             num_actions,
#             actor_hidden_dims,
#             critic_hidden_dims,
#             activation,
#             init_noise_std,
#             noise_std_type,
#             **kwargs
#         )

#         # 获取一帧obs的长度
#         self.one_obs_len: int = int(num_actor_obs / num_history_len)

#         activation = resolve_nn_activation(activation)

#         # Policy 构建actor网络
#         actor_layers = []
#         actor_layers.append(nn.Linear(self.one_obs_len + num_latent, actor_hidden_dims[0]))
#         actor_layers.append(activation)
#         for layer_index in range(len(actor_hidden_dims)):
#             if layer_index == len(actor_hidden_dims) - 1:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
#             else:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
#                 actor_layers.append(activation)
#         self.actor = nn.Sequential(*actor_layers)

#         # 构建encoder网络
#         encoder_layers = []
#         encoder_layers.append(nn.Linear(num_actor_obs, encoder_hidden_dims[0]))
#         encoder_layers.append(activation)
#         for layer_index in range(len(encoder_hidden_dims) - 1):
#             encoder_layers.append(nn.Linear(encoder_hidden_dims[layer_index], encoder_hidden_dims[layer_index + 1]))
#             encoder_layers.append(activation)
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.encode_latent = nn.Linear(encoder_hidden_dims[-1],num_latent-3) # 输出隐向量的均值
#         self.encode_vel = nn.Linear(encoder_hidden_dims[-1],3) # 输出速度的均值

#         # 输出网络结构
#         print(f"Actor MLP: {self.actor}")
#         print(f"Encoder MLP: {self.encoder}")


#     def encoder_forward(self,obs_history):
#         """EstNet 前向推理
#         Args:
#             obs_history (_type_): 历史观测值

#         """
#         x = self.encoder(obs_history)
#         latent = self.encode_latent(x) # 得到隐向量
#         est_vel = self.encode_vel(x) # 得到速度估计值

#         code = torch.cat((est_vel,latent),dim=-1)
#         # code = est_vel
#         return code,est_vel

#     def act(self, obs_history, **kwargs):
#         """训练时使用的前向推理函数,action经过正态分布采样再输出

#         Args:
#             obs_history (_type_): 当前观测值
#             obs_history (_type_): 观测值历史
#         """
#         code,_ = self.encoder_forward(obs_history)
#         now_obs = obs_history[:, -self.one_obs_len:]

#         # TODO:对critic进行检查，real_vel不正确设置就报错
#         # 实现adaboot
#         critic_obs = kwargs.get("critic_obs", None)
#         # rewards = kwargs.get("rewards", None)
#         real_vel = critic_obs[:, 0:3]
#         # if critic_obs is not None and rewards is not None:
#         #     real_vel = critic_obs[:, 0:3]
#         #     CV_R = torch.std(rewards) / torch.mean(rewards)
#         #     p_boot = 1 - torch.tanh(CV_R)
#         #     random_tensor = torch.rand(real_vel.shape, device=rewards.device)
#         #     code[:, 0:3] = torch.where(random_tensor < p_boot.expand_as(random_tensor), real_vel, code[:, 0:3])
#         #     # 完全使用真实值，用于debug
#         #     # code[:, 0:3] = real_vel
        
#         observations = torch.cat((code.detach(),now_obs),dim=-1) # 隐向量放在当前观测值前
#         self.update_distribution(observations)
#         return self.distribution.sample()


#     def act_inference(self, obs_history):
#         """部署时使用的前向推理函数

#         Args:
#             obs_history (_type_): _description_
#             obs_history (_type_): _description_
#         """

#         code,_ = self.encoder_forward(obs_history)
#         now_obs = obs_history[:, -self.one_obs_len:]
#         observations = torch.cat((code.detach(),now_obs),dim=-1)
#         actions_mean = self.actor(observations)
#         return actions_mean