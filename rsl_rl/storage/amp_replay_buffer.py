# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-resident replay of AMP state transitions."""

from __future__ import annotations

import torch


class AMPReplayBuffer:
    """Store raw, detached state pairs independently of PPO rollout storage."""

    def __init__(self, observation_dim: int, capacity: int, device: str = "cpu") -> None:
        """Allocate transition storage on the selected device."""
        if observation_dim < 1 or capacity < 1:
            raise ValueError("AMP observation_dim and replay capacity must be positive.")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.states = torch.empty(capacity, observation_dim, dtype=torch.float32, device=device)
        self.next_states = torch.empty_like(self.states)
        self.device = self.states.device
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        """Return the number of initialized transition pairs."""
        return self.size

    @torch.no_grad()
    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Copy transitions into a ring, retaining the newest entries on overflow."""
        if states.ndim != 2 or states.shape != next_states.shape or states.shape[1] != self.observation_dim:
            raise ValueError("AMP replay expects two tensors with shape [batch, observation_dim].")
        count = states.shape[0]
        if count == 0:
            return
        if count >= self.capacity:
            self.states.copy_(states[-self.capacity :].detach())
            self.next_states.copy_(next_states[-self.capacity :].detach())
            self.position = 0
            self.size = self.capacity
            return
        first = min(count, self.capacity - self.position)
        self.states[self.position : self.position + first].copy_(states[:first].detach())
        self.next_states[self.position : self.position + first].copy_(next_states[:first].detach())
        remaining = count - first
        if remaining:
            self.states[:remaining].copy_(states[first:].detach())
            self.next_states[:remaining].copy_(next_states[first:].detach())
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample paired states uniformly with replacement."""
        if batch_size < 1:
            raise ValueError("AMP replay batch_size must be positive.")
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty AMP replay buffer.")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return self.states[indices], self.next_states[indices]

    def clear(self) -> None:
        """Reset the replay cursor without reallocating device memory."""
        self.size = 0
        self.position = 0
