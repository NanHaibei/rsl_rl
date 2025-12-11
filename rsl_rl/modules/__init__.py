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

# ElevationNet: 三个独立的mode实现
from .actor_critic_ElevationNet_mode1 import ActorCriticElevationNetMode1
from .actor_critic_ElevationNet_mode2 import ActorCriticElevationNetMode2
from .actor_critic_ElevationNet_mode3 import ActorCriticElevationNetMode3

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
    "ActorCriticElevationNetMode1",
    "ActorCriticElevationNetMode2",
    "ActorCriticElevationNetMode3",
]
