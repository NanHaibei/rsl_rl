# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import Normalizer
from rsl_rl.networks import MLP, EmpiricalNormalization

class AMPDiscriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the input feature vector (concatenated state and next state).
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0, use_normalize=True):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        # amp_layers = []
        # curr_in_dim = input_dim
        # for hidden_dim in hidden_layer_sizes:
        #     amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
        #     amp_layers.append(nn.ReLU())
        #     curr_in_dim = hidden_dim
        # self.trunk = nn.Sequential(*amp_layers).to(device)
        # self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        # self.trunk.train()
        # self.amp_linear.train()

        self.net = MLP(input_dim, 1, hidden_layer_sizes).to(device).train()

        print(f"Discriminator: {self.net}")

        self.task_reward_lerp = task_reward_lerp
        self.use_normalize = use_normalize
        if self.use_normalize:
            self.normalizer = EmpiricalNormalization(int(input_dim / 2))
            print("Discriminator: Using input normalization")
        else:
            self.normalizer = None
            print("Discriminator: Normalization disabled")

    def forward(self, now_state, next_state, update_normalizer=True):
        """
        推理判别器
        
        Args:
            now_state (torch.Tensor): 当前状态观测
            next_state (torch.Tensor): 下一状态观测
            update_normalizer (bool): 是否更新归一化器统计量。
                                     对于策略数据应该设为 True，
                                     对于专家数据应该设为 False。
        """
        # h = self.trunk(x)
        # d = self.amp_linear(h)
        
        # 如果使用归一化
        if self.use_normalize:
            # 只在明确要求且处于训练模式时更新归一化器参数
            # 策略数据应该更新 normalizer，专家数据不应该更新
            if update_normalizer and self.training:
                self.normalizer.update(now_state)
                self.normalizer.update(next_state)
            
            # 执行归一化
            now_state = self.normalizer(now_state)
            next_state = self.normalizer(next_state)
        
        # 拼接当前帧与下一帧
        input = torch.cat([now_state, next_state], dim=-1)
        # 推理并返回
        return self.net(input)

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        
        # 如果使用归一化，手动归一化（不更新 normalizer 统计量）
        if self.use_normalize:
            with torch.no_grad():
                expert_state_norm = self.normalizer(expert_state)
                expert_next_state_norm = self.normalizer(expert_next_state)
                # 拼接归一化后的数据
                expert_data = torch.cat([expert_state_norm, expert_next_state_norm], dim=-1)
        else:
            # 不使用归一化，直接拼接
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        
        # 对拼接后的数据启用梯度追踪（用于梯度惩罚计算）
        expert_data = expert_data.detach().requires_grad_(True)
        
        # 直接通过网络（跳过 forward 方法，避免重复归一化）
        disc = self.net(expert_data)
        
        ones = torch.ones(disc.size(), device=disc.device)
        
        # 对拼接后的数据计算梯度
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, 
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()

            # 调用 forward 方法，但不更新 normalizer（推理阶段不应更新统计量）
            d = self.forward(state, next_state, update_normalizer=False)
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                final_reward = self._lerp_reward(style_reward, task_reward.unsqueeze(-1))
            else:
                final_reward = style_reward
            self.train()
        return final_reward.squeeze(), style_reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
