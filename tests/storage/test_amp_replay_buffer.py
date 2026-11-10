# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMP transition replay."""

import torch

import pytest

from rsl_rl.storage import AMPReplayBuffer


def test_replay_wrap_oversized_insert_and_pair_alignment() -> None:
    """Preserve state-pair correspondence through wrapping and oversized batches."""
    buffer = AMPReplayBuffer(1, 5)
    first = torch.arange(3.0).unsqueeze(1)
    buffer.insert(first, first + 10)
    second = torch.arange(3.0, 7.0).unsqueeze(1)
    buffer.insert(second, second + 10)
    assert len(buffer) == 5
    assert buffer.position == 2
    assert set(buffer.states.flatten().tolist()) == {2, 3, 4, 5, 6}
    large = torch.arange(100.0, 107.0).unsqueeze(1)
    buffer.insert(large, large + 10)
    assert set(buffer.states.flatten().tolist()) == {102, 103, 104, 105, 106}
    states, next_states = buffer.sample(50)
    torch.testing.assert_close(next_states, states + 10)
    buffer.clear()
    with pytest.raises(RuntimeError, match="empty"):
        buffer.sample(1)


def test_replay_owns_detached_copies_and_handles_empty_insert() -> None:
    """Own replay memory and keep inference tensors out of training graphs."""
    buffer = AMPReplayBuffer(2, 8)
    value = torch.ones(3, 2, requires_grad=True)
    with torch.inference_mode():
        buffer.insert(value, value + 1)
    with torch.no_grad():
        value.zero_()
    assert not buffer.states.requires_grad
    torch.testing.assert_close(buffer.states[:3], torch.ones(3, 2))
    buffer.insert(torch.empty(0, 2), torch.empty(0, 2))
    assert len(buffer) == 3
    with pytest.raises(ValueError, match="shape"):
        buffer.insert(torch.ones(3, 1), torch.ones(3, 1))
