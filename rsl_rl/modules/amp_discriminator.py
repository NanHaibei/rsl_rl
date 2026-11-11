# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Least-squares AMP discriminator and shared state normalization."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.utils import reduce_gradients_in_buckets

from .mlp import MLP
from .normalization import EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """Score consecutive states and produce bounded style rewards."""

    def __init__(
        self,
        observation_dim: int,
        amp_cfg: dict,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Initialize the discriminator, optimizer, replay buffers, and update schedule."""
        from rsl_rl.storage import AMPReplayBuffer

        super().__init__()
        self.cfg = dict(amp_cfg)
        self.device = device
        self.observation_dim = observation_dim
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb

        self.network = MLP(
            2 * observation_dim, 1, amp_cfg["discriminator_hidden_dims"], amp_cfg["discriminator_activation"]
        )
        self.normalizer = (
            EmpiricalNormalization(observation_dim, eps=1e-4)
            if amp_cfg["discriminator_obs_normalization"]
            else nn.Identity()
        )
        self.to(device)
        head_parameters = list(self.network[-1].parameters())
        head_parameter_ids = {id(parameter) for parameter in head_parameters}
        trunk_parameters = [
            parameter for parameter in self.network.parameters() if id(parameter) not in head_parameter_ids
        ]
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": trunk_parameters,
                    "weight_decay": amp_cfg.get("discriminator_trunk_weight_decay", 0.0),
                    "name": "amp_trunk",
                },
                {
                    "params": head_parameters,
                    "weight_decay": amp_cfg.get("discriminator_head_weight_decay", 0.0),
                    "name": "amp_head",
                },
            ],
            lr=amp_cfg["discriminator_learning_rate"],
        )

        self.policy_buffer = AMPReplayBuffer(observation_dim, amp_cfg["replay_buffer_size"], device)
        self.expert_buffer = AMPReplayBuffer(observation_dim, amp_cfg["replay_buffer_size"], device)
        self.update_counter = 0

    def sync_learning_rate(self, policy_learning_rate: float) -> None:
        """Apply PPO's current learning rate when policy following is enabled."""
        if self.cfg.get("discriminator_follow_policy_lr", False):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = policy_learning_rate

    def load_optimizer_state_dict(self, state_dict: dict) -> None:
        """Restore Adam state while applying the configured weight decay."""
        self.optimizer.load_state_dict(state_dict)
        self.optimizer.param_groups[0]["weight_decay"] = self.cfg.get("discriminator_trunk_weight_decay", 0.0)
        self.optimizer.param_groups[1]["weight_decay"] = self.cfg.get("discriminator_head_weight_decay", 0.0)

    @torch.no_grad()
    def process_env_step(
        self,
        prev_obs: TensorDict,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        *,
        amp_obs_group: str,
        amp_expert_obs_group: str,
    ) -> tuple[torch.Tensor, TensorDict | None]:
        """Record policy transitions and independently sampled expert pairs."""
        # 这里的next_amp_obs是用于构造actor状态对的
        amp_obs = prev_obs[amp_obs_group]
        next_amp_obs = obs[amp_obs_group].clone()
        expert_pairs = obs[amp_expert_obs_group]
        done_mask = dones.bool()

        terminal_obs = None
        if done_mask.any():
            terminal_obs = TensorDict(extras["final_obs"], batch_size=obs.batch_size, device=self.device)
            next_amp_obs[done_mask] = terminal_obs[amp_obs_group][done_mask]

        self.policy_buffer.insert(amp_obs, next_amp_obs)
        self.expert_buffer.insert(expert_pairs[:, 0], expert_pairs[:, 1])

        task_rewards = rewards.clone()
        style_rewards = self.predict_reward(amp_obs, next_amp_obs, self.cfg["reward_coef"])
        lerp = self.cfg["task_reward_lerp"]
        final_rewards = lerp * task_rewards + (1.0 - lerp) * style_rewards
        extras["amp_rewards"] = {"task": task_rewards, "style": style_rewards, "final": final_rewards}

        return final_rewards, terminal_obs

    def update(self, batch_size: int) -> torch.Tensor | None:
        """Update from replay on schedule, returning losses when an update runs."""
        self.update_counter += 1
        # 检查是否该更新判别器了
        if (self.update_counter - 1) % self.cfg["discriminator_update_interval"]:
            return None

        # 汇总各卡的 ready,决定是否更新,所有卡必须一起更新或一起跳过，保证后续通信一致
        ready = len(self.policy_buffer) > 0 and len(self.expert_buffer) > 0
        if self.is_multi_gpu:
            ready_tensor = torch.tensor(int(ready), device=self.device)
            torch.distributed.all_reduce(ready_tensor, op=torch.distributed.ReduceOp.MIN)
            ready = bool(ready_tensor.item())
        if not ready:
            return None

        policy = self.policy_buffer.sample(batch_size)
        expert = self.expert_buffer.sample(batch_size)
        with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
            values = self.compute_loss(*policy, *expert, self.cfg["gradient_penalty_coef"])
        loss = values[0]

        self.optimizer.zero_grad()
        loss.backward()
        # 让判别器共同学习所有卡的数据，并保持参数一致
        if self.is_multi_gpu:
            reduce_gradients_in_buckets(self.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb)
        nn.utils.clip_grad_norm_(self.parameters(), self.cfg["discriminator_max_grad_norm"], error_if_nonfinite=True)
        self.optimizer.step()

        statistics = torch.stack([value.detach() for value in values])
        # 让日志反映所有卡的训练情况
        if self.is_multi_gpu:
            torch.distributed.all_reduce(statistics)
            statistics /= self.gpu_world_size
        return statistics

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        """Score normalized transition features."""
        return self.network(transitions).squeeze(-1)

    @torch.no_grad()
    def predict_reward(self, states: torch.Tensor, next_states: torch.Tensor, reward_coef: float) -> torch.Tensor:
        """Return style rewards bounded between zero and reward_coef."""
        score = self(torch.cat((self.normalizer(states), self.normalizer(next_states)), dim=-1))
        return reward_coef * (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)

    def compute_loss(
        self,
        policy_states: torch.Tensor,
        policy_next_states: torch.Tensor,
        expert_states: torch.Tensor,
        expert_next_states: torch.Tensor,
        gradient_penalty_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute least-squares loss and a zero-centered penalty on normalized expert inputs."""
        if isinstance(self.normalizer, EmpiricalNormalization):
            with torch.no_grad():
                self.normalizer.update(
                    torch.cat((policy_states, policy_next_states, expert_states, expert_next_states)).detach()
                )
        policy_input = torch.cat(
            (self.normalizer(policy_states), self.normalizer(policy_next_states)), dim=-1
        ).detach()
        expert_input = torch.cat(
            (self.normalizer(expert_states), self.normalizer(expert_next_states)), dim=-1
        ).detach()
        expert_input.requires_grad_(True)
        policy_score = self(policy_input)
        expert_score = self(expert_input)
        discriminator_loss = 0.5 * ((policy_score + 1.0).square().mean() + (expert_score - 1.0).square().mean())
        gradient = torch.autograd.grad(expert_score.sum(), expert_input, create_graph=True)[0]
        gradient_penalty = gradient.square().sum(dim=-1).mean()
        loss = discriminator_loss + gradient_penalty_coef * gradient_penalty
        return loss, discriminator_loss, gradient_penalty, policy_score.detach().mean(), expert_score.detach().mean()
