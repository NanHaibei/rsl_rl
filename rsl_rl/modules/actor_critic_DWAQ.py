# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic_DWAQ(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_encoder_obs,
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
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # 返回已经实例化的激活函数类
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs + num_latent # actor网络输入维度是观测值和隐向量的拼接
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
        encoder_layers.append(nn.Linear(num_encoder_obs, encoder_hidden_dims[0]))
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

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
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

    def update_distribution(self, observations):
        # compute mean actor网络推理得到正态分布均值
        mean = self.actor(observations)
        # compute standard deviation 
        if self.noise_std_type == "scalar":
            # 复制一个维度是mean的张量，内容是std的值
            std = self.std.expand_as(mean) 
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution 用均值和标准差创建正态分布
        # 创建的其实是一个批量分布，维度和mean、std相同
        self.distribution = Normal(mean, std)

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
        mean_vel = self.encode_mean_vel(x) # 得到速度的均值
        logvar_vel = self.encode_logvar_vel(x) # 得到速度的对数方差

        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

    def act(self, observations, obs_history, **kwargs):
        """训练时使用的前向推理函数,policy输出经过正态分布再

        Args:
            observations (_type_): 当前观测值
            obs_history (_type_): 观测值历史
        """
        code,_,_,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code.detach(),observations),dim=-1) # 隐向量放在当前观测值前面
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """计算动作值的概率分布

        Args:
            actions (_type_): 动作值
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        """部署时使用的前向推理函数

        Args:
            observations (_type_): _description_
            obs_history (_type_): _description_
        """

        code,_,_,_,_,_,_ = self.cenet_forward(obs_history)
        # x = self.encoder(obs_history)
        # mean_vel = self.encode_mean_vel(x)
        # mean_latent = self.encode_mean_latent(x)
        # code = torch.cat((mean_vel,mean_latent),dim=-1)
        observations = torch.cat((code.detach(),observations),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """critic网络输出值函数

        Args:
            critic_observations (_type_): critic的观测值
        """
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
