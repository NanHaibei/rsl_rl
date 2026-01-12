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
from .actor_critic_ElevationNet_mode1 import ActorCriticElevationNetMode1
from .actor_critic_ElevationNet_mode2 import ActorCriticElevationNetMode2
from .actor_critic_ElevationNet_mode2A import ActorCriticElevationNetMode2A
from .actor_critic_ElevationNet_mode3 import ActorCriticElevationNetMode3
from .actor_critic_ElevationNet_mode4 import ActorCriticElevationNetMode4
from .actor_critic_ElevationNet_mode5 import ActorCriticElevationNetMode5
from .actor_critic_ElevationNet_mode6 import ActorCriticElevationNetMode6
from .actor_critic_ElevationNet_mode7 import ActorCriticElevationNetMode7
from .actor_critic_ElevationNet_mode8 import ActorCriticElevationNetMode8
from .actor_critic_ElevationNet_mode9 import ActorCriticElevationNetMode9
from .actor_critic_ElevationNet_mode9A import ActorCriticElevationNetMode9A
from .actor_critic_ElevationNet_mode10 import ActorCriticElevationNetMode10
from .actor_critic_ElevationNet_mode11 import ActorCriticElevationNetMode11
from .actor_critic_ElevationNet_mode12P2 import ActorCriticElevationNetMode12P2
from .actor_critic_ElevationNet_mode12L import ActorCriticElevationNetMode12L
# from .actor_critic_ElevationNet_mode12P2 import ActorCriticElevationNetMode12

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
    "ActorCriticElevationNetMode2A",
    "ActorCriticElevationNetMode3",
    "ActorCriticElevationNetMode4",
    "ActorCriticElevationNetMode5",
    "ActorCriticElevationNetMode6",
    "ActorCriticElevationNetMode7",
    "ActorCriticElevationNetMode8",
    "ActorCriticElevationNetMode9",
    "ActorCriticElevationNetMode9A",
    "ActorCriticElevationNetMode10",
    "ActorCriticElevationNetMode11",
    "ActorCriticElevationNetMode12",
    "ActorCriticElevationNetMode12L",
    "ActorCriticElevationNetMode12P2",
]
