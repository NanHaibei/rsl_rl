# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic_DeltaSine(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        env,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        deltasine_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        num_history_len = 5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic_DeltaSine.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # 返回已经实例化的激活函数类
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy 构建actor网络
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a - 4 + 4, actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # 历史观测值缓冲区初始化标签
        # self.history_obs_flag = False
        # self.num_history_len = num_history_len
        # self.history_obs = torch.zeros(4096,mlp_input_dim_a * num_history_len)
        # self.env = env
        # 创建DeltaSine网络，输入是历史观测值，输出是fre、rate、offset的增量
        deltasine_layers = []
        deltasine_layers.append(nn.Linear(mlp_input_dim_a - 4, deltasine_hidden_dims[0]))
        deltasine_layers.append(activation)
        for layer_index in range(len(deltasine_hidden_dims)):
            if layer_index == len(deltasine_hidden_dims) - 1:
                deltasine_layers.append(nn.Linear(deltasine_hidden_dims[layer_index], 3))
            else:
                deltasine_layers.append(nn.Linear(deltasine_hidden_dims[layer_index], deltasine_hidden_dims[layer_index + 1]))
                deltasine_layers.append(activation)
        self.deltasine = nn.Sequential(*deltasine_layers)

        # 输出三个网络的结构
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"DeltaSine MLP: {self.deltasine}")

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

    def act(self, observations, **kwargs):
        # 初始化历史观测值缓冲区
        # if self.history_obs_flag == False or self.history_obs.shape[0] != observations.shape[0]:
        #     self.history_obs = observations.repeat(1,self.num_history_len) 
        #     self.history_obs_flag = True
        # 输出步态信息的增量
        # self.history_obs = torch.roll(self.history_obs,shifts=observations.shape[1],dims=1) # 向右滚动一帧
        # self.history_obs[:,0:observations.shape[1]] = observations # 将最新的观测值放在最前面
        delta_signal = self.deltasine(observations[:, 0:45]) 
        
        stance_rate = observations[:, 45] + delta_signal[:,0] # stance_rate
        bipedal_offset = observations[:, 46] + delta_signal[:,1] # bipedal_offset
        gait_frequency = observations[:, 47] + delta_signal[:,2] # gait_frequency

        # 生成正余弦信号
        phase = (observations[:, 48] * gait_frequency) % 1.0
        left_sin = torch.sin(2 * torch.pi * phase)
        left_cos = torch.cos(2 * torch.pi * phase)
        right_sin = torch.sin(2 * torch.pi * (phase + bipedal_offset))
        right_cos = torch.cos(2 * torch.pi * (phase + bipedal_offset))

        # 拼接正余弦信号到观测值中
        policy_input = torch.cat(
            (
                observations[:,0:45],
                left_sin.unsqueeze(1),
                left_cos.unsqueeze(1),
                right_sin.unsqueeze(1),
                right_cos.unsqueeze(1)
            ),
            dim=-1
        )

        self.update_distribution(policy_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        

        
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
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
    
    def _init_nn(self):
        super().__init__()
