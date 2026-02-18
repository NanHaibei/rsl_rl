# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
import copy
import os

class ActorCriticMode13A1(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg=None,
        alg_cfg: dict | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        vision_spatial_size: tuple[int, int] = (25, 17),
        elevation_history_length: int = 5,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()
        
        # 保存高程图配置
        self.vision_spatial_size = vision_spatial_size
        self.elevation_history_length = elevation_history_length

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == "height_scan_policy":
                continue  # 跳过高程图，单独计算
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if obs_group == "height_scan_critic":
                continue  # 跳过高程图，单独计算
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        # 计算高程图flatten后的维度（包含历史帧）
        # 高程图格式: [B, history_length, H, W] -> flatten后: [B, history_length * H * W]
        elevation_flat_dim = elevation_history_length * vision_spatial_size[0] * vision_spatial_size[1]
        
        # Actor输入 = 本体观测 + flatten的高程图历史
        actor_input_dim = num_actor_obs + elevation_flat_dim
        
        # Critic输入 = 本体特权观测 + flatten的高程图历史
        critic_input_dim = num_critic_obs + elevation_flat_dim

        self.state_dependent_std = state_dependent_std

        # Actor (输入维度改变，网络结构不变)
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")
        print(f"  Input: {actor_input_dim} (proprio: {num_actor_obs} + elevation: {elevation_flat_dim})")
        print(f"  Elevation: {elevation_history_length} frames x {vision_spatial_size[0]}x{vision_spatial_size[1]}")

        # Actor observation normalization (只对本体观测归一化)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic (输入维度改变，网络结构不变)
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")
        print(f"  Input: {critic_input_dim} (proprio: {num_critic_obs} + elevation: {elevation_flat_dim})")
        print(f"  Elevation: {elevation_history_length} frames x {vision_spatial_size[0]}x{vision_spatial_size[1]}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)
    
    def _normalize_elevation_map(self, height_map: torch.Tensor) -> torch.Tensor:
        """归一化高程图
        
        Args:
            height_map: 高程图，形状为 [B, history_length, H, W]
        
        Returns:
            归一化后的高程图，形状为 [B, history_length, H, W]
        """
        # height_map: [B, history_length, H, W]
        # 对每帧的空间维度进行归一化
        height_map_mean = height_map.mean(dim=(-2, -1), keepdim=True)  # [B, history_length, 1, 1]
        height_map = torch.clip((height_map - height_map_mean) / 0.6, -3.0, 3.0)
        return height_map

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # 提取并处理高程图: [B, history_length, H, W]
        height_map = obs["height_scan_policy"]
        height_map = self._normalize_elevation_map(height_map)
        height_map_flat = height_map.flatten(1)  # [B, history_length, H, W] -> [B, history_length*H*W]
        
        # 提取并归一化本体观测
        proprio_obs = obs["policy"]
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 直接concat
        fused_obs = torch.cat((proprio_obs, height_map_flat), dim=-1)
        
        self._update_distribution(fused_obs)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        # 提取并处理高程图: [B, history_length, H, W]
        height_map = obs["height_scan_policy"]
        height_map = self._normalize_elevation_map(height_map)
        height_map_flat = height_map.flatten(1)  # [B, history_length, H, W] -> [B, history_length*H*W]
        
        # 提取并归一化本体观测
        proprio_obs = obs["policy"]
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 直接concat
        fused_obs = torch.cat((proprio_obs, height_map_flat), dim=-1)
        
        if self.state_dependent_std:
            return self.actor(fused_obs)[..., 0, :], self.extra_info
        else:
            return self.actor(fused_obs), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # 提取并处理高程图: [B, history_length, H, W]
        height_map = obs["height_scan_critic"]
        height_map = self._normalize_elevation_map(height_map)
        height_map_flat = height_map.flatten(1)  # [B, history_length, H, W] -> [B, history_length*H*W]
        
        # 提取并归一化本体观测
        proprio_obs = obs["critic"]
        proprio_obs = self.critic_obs_normalizer(proprio_obs)
        
        # 直接concat
        fused_obs = torch.cat((proprio_obs, height_map_flat), dim=-1)
        
        return self.critic(fused_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group == "height_scan_policy":
                continue
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            if obs_group == "height_scan_critic":
                continue
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["critic"][0]].shape[0], 0)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典，包含主要的优化器
        """
        import torch.optim as optim
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return {"optimizer": optimizer}

    def export_to_onnx(self, path: str, filename: str = "policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建导出器
        exporter = _OnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _OnnxPolicyExporter(torch.nn.Module):
    """Mode13A1的ONNX导出器"""
    
    def __init__(self, policy: ActorCriticMode13A1, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # 复制策略参数
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        # 获取维度信息
        self.vision_spatial_size = policy.vision_spatial_size
        self.elevation_history_length = policy.elevation_history_length
        elevation_flat_dim = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
        self.proprio_obs_dim = policy.actor[0].in_features - elevation_flat_dim

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
    
    def _normalize_elevation_map(self, height_map: torch.Tensor) -> torch.Tensor:
        """归一化高程图
        
        Args:
            height_map: 高程图，形状为 [B, history_length, H, W]
        
        Returns:
            归一化后的高程图，形状为 [B, history_length, H, W]
        """
        # height_map: [B, history_length, H, W]
        # 对每帧的空间维度进行归一化
        height_map_mean = height_map.mean(dim=(-2, -1), keepdim=True)  # [B, history_length, 1, 1]
        height_map = torch.clip((height_map - height_map_mean) / 0.6, -3.0, 3.0)
        return height_map
    
    def forward(self, obs_concat):
        """输入: [batch, proprio_dim + history_length*H*W], 输出: [batch, num_actions]"""
        # 分离本体观测和高程图
        proprio_obs = obs_concat[:, :self.proprio_obs_dim]
        elevation_flat = obs_concat[:, self.proprio_obs_dim:]
        
        # 归一化本体观测
        proprio_obs = self.normalizer(proprio_obs)
        
        # 归一化高程图
        # 将展平的高程图 reshape 回 [B, history_length, H, W]
        height, width = self.vision_spatial_size
        batch_size = elevation_flat.shape[0]
        elevation_map = elevation_flat.reshape(batch_size, self.elevation_history_length, height, width)
        # 应用归一化
        elevation_map = self._normalize_elevation_map(elevation_map)
        # 重新展平
        elevation_flat = elevation_map.flatten(1)
        
        # concat并输出
        fused_obs = torch.cat([proprio_obs, elevation_flat], dim=-1)
        actions_mean = self.actor(fused_obs)
        return actions_mean
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        
        height, width = self.vision_spatial_size
        history_length = self.elevation_history_length
        elevation_dim = history_length * height * width
        total_dim = self.proprio_obs_dim + elevation_dim
        obs_concat = torch.zeros(1, total_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (Mode13A1):")
        print(f"{'='*80}")
        print(f"  本体观测维度:     {self.proprio_obs_dim}")
        print(f"  高程图尺寸:       {history_length} x {height} x {width}")
        print(f"  高程图展平维度:   {elevation_dim}")
        print(f"  总输入维度:       {total_dim}")
        print(f"  输入格式: [本体观测({self.proprio_obs_dim}) + 高程图展平({elevation_dim})]")
        print(f"{'='*80}\n")
        
        torch.onnx.export(
            self,
            obs_concat,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs_concat"],
            output_names=["actions"],
            dynamic_axes={
                "obs_concat": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )
