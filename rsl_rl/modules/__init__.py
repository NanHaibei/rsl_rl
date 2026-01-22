# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import resolve_symmetry_config

from .actor_critic_EstNet import ActorCriticEstNet
from .actor_critic_DWAQ import ActorCriticDWAQ
from .amp_discriminator import AMPDiscriminator

# ElevationNet: 九个独立的mode实现
from .actor_critic_ElevationNet_mode12P2 import ActorCriticElevationNetMode12P2
from .actor_critic_ElevationNet_mode12L import ActorCriticElevationNetMode12L

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "resolve_rnd_config",
    "resolve_symmetry_config",

    "ActorCriticEstNet",
    "ActorCriticDWAQ",
    "AMPDiscriminator",
    
    # ElevationNet新的独立实现
    "ActorCriticElevationNetMode12L",
    "ActorCriticElevationNetMode12P2",
]
