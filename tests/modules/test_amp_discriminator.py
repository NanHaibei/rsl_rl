# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMP rewards, gradients, and normalization."""

import torch
from tensordict import TensorDict

from rsl_rl.modules import AMPDiscriminator


def _make_discriminator(obs_normalization: bool = True, interval: int = 1) -> AMPDiscriminator:
    return AMPDiscriminator(
        observation_dim=2,
        amp_cfg={
            "discriminator_hidden_dims": [4],
            "discriminator_activation": "relu",
            "discriminator_obs_normalization": obs_normalization,
            "discriminator_learning_rate": 1e-3,
            "discriminator_max_grad_norm": 1.0,
            "gradient_penalty_coef": 10.0,
            "replay_buffer_size": 16,
            "discriminator_update_interval": interval,
        },
    )


def test_style_reward_bounds_and_least_squares_targets() -> None:
    """Match the analytic least-squares targets and bounded style reward."""
    discriminator = _make_discriminator(obs_normalization=False)
    for parameter in discriminator.parameters():
        parameter.data.zero_()
    states = torch.ones(3, 2)
    values = discriminator.compute_loss(states, states, states, states, 10.0)
    torch.testing.assert_close(values[1], torch.tensor(1.0))
    torch.testing.assert_close(values[2], torch.tensor(0.0))
    torch.testing.assert_close(discriminator.predict_reward(states, states, 2.0), torch.full((3,), 1.5))
    for bias, reward in ((1.0, 2.0), (-1.0, 0.0), (3.0, 0.0)):
        discriminator.network[-1].bias.data.fill_(bias)
        torch.testing.assert_close(discriminator.predict_reward(states, states, 2.0), torch.full((3,), reward))


def test_gradient_penalty_uses_normalized_inputs_and_detaches_samples() -> None:
    """Differentiate normalized expert features without linking rollout graphs."""
    discriminator = _make_discriminator()
    discriminator.normalizer.eval()
    discriminator.normalizer._std.fill_(2.0)
    with torch.no_grad():
        discriminator.network[0].weight.copy_(torch.eye(4))
        discriminator.network[0].bias.fill_(100.0)
        discriminator.network[-1].weight.fill_(1.0)
        discriminator.network[-1].bias.zero_()
    states = torch.ones(3, 2, requires_grad=True)
    values = discriminator.compute_loss(states, states, states, states, 10.0)
    torch.testing.assert_close(values[2], torch.tensor(4.0))
    torch.testing.assert_close(values[0], values[1] + 40.0)
    values[0].backward()
    assert states.grad is None
    assert all(torch.isfinite(parameter.grad).all() for parameter in discriminator.parameters())


def test_process_env_step_supports_custom_groups_without_gradients() -> None:
    """Collect transitions and mix rewards independently of PPO with custom observation groups."""
    discriminator = _make_discriminator(obs_normalization=False)
    discriminator.cfg.update(reward_coef=2.0, task_reward_lerp=0.25)
    with torch.no_grad():
        for parameter in discriminator.parameters():
            parameter.zero_()
        discriminator.network[-1].bias.fill_(1.0)
    states = torch.ones(2, 2, requires_grad=True)
    prev_obs = TensorDict({"amp_state": states}, batch_size=[2])
    obs = TensorDict({"amp_state": states + 1.0, "expert_pairs": torch.randn(2, 2, 2)}, batch_size=[2])
    rewards = torch.tensor([2.0, 6.0], requires_grad=True)
    extras = {}
    mixed, terminal_obs = discriminator.process_env_step(
        prev_obs,
        obs,
        rewards,
        torch.zeros(2, dtype=torch.bool),
        extras,
        amp_obs_group="amp_state",
        amp_expert_obs_group="expert_pairs",
    )
    assert terminal_obs is None
    torch.testing.assert_close(mixed, torch.tensor([2.0, 3.0]))
    assert not mixed.requires_grad
    assert all(not component.requires_grad for component in extras["amp_rewards"].values())
    assert len(discriminator.policy_buffer) == len(discriminator.expert_buffer) == 2


def test_update_owns_replay_optimizer_and_schedule() -> None:
    """Train independently of PPO after both replay sources become available."""
    discriminator = _make_discriminator(interval=3)
    assert discriminator.update(4) is None
    states = torch.randn(8, 2)
    discriminator.policy_buffer.insert(states, states + 0.1)
    discriminator.expert_buffer.insert(states + 0.5, states + 0.6)
    before = [parameter.detach().clone() for parameter in discriminator.parameters()]
    assert discriminator.update(4) is None
    assert discriminator.update(4) is None
    statistics = discriminator.update(4)
    assert discriminator.update_counter == 4
    assert statistics.shape == (3,)
    assert torch.isfinite(statistics).all()
    assert not statistics.requires_grad
    assert discriminator.optimizer.state
    assert any(not torch.equal(left, right) for left, right in zip(before, discriminator.parameters()))
