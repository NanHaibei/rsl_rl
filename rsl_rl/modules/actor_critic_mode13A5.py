# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticMode13A5: Actor和Critic都用2DCNN处理多帧高程图历史（作为通道）

网络结构:
    Critic Pipeline:
    - 单帧本体特权观测 + 多帧高程图历史(2DCNN提取特征) -> concat -> Critic网络 -> Value
    
    Actor Pipeline:
    - 多帧高程图历史 -> 2DCNN提取特征
    - 本体观测 -> MLP提取特征
    - 本体特征 + 视觉特征 -> concat -> Actor网络 -> Actions
    
    相比Mode13A4的改进：
    - 高程图输入从单帧改为多帧历史（如5帧）
    - 历史帧作为通道维度处理：输入 [B, T, H, W] 而非 [B, H, W]
    - CNN自动学习时序特征
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
import copy
import os


class Elevation2DCNNEncoder(nn.Module):
    """2DCNN编码器，用于处理多帧高程图历史（作为通道）"""
    
    def __init__(self, 
                 in_channels=1,  # 历史帧数，如5
                 hidden_dims=[16, 32, 64],
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2],
                 out_dim=64,
                 vision_spatial_size=(25, 17)):
        super().__init__()
        
        # 构建2DCNN卷积层
        layers = []
        now_channels = in_channels
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            layers.append(nn.Conv2d(
                now_channels, 
                hidden_dim, 
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2
            ))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            now_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过卷积后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W] 多帧高程图历史（已归一化），T为历史帧数
        Returns:
            features: [B, out_dim] 特征向量
        """
        # 不需要添加通道维度，输入已经是 [B, T, H, W] 格式
        # 直接通过2DCNN，CNN会处理T个通道
        
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ActorCriticMode13A5(nn.Module):
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
        # 高程图编码器配置
        vision_spatial_size: tuple[int, int] = (25, 17),
        vision_feature_dim: int = 64,
        # 高程图历史长度
        elevation_history_length: int = 5,
        # Actor 2DCNN配置
        actor_cnn_hidden_dims: list[int] = [16, 32, 64],
        actor_cnn_kernel_sizes: list[int] = [3, 3, 3],
        actor_cnn_strides: list[int] = [2, 2, 2],
        # Critic 2DCNN配置（默认使用相同配置）
        critic_cnn_hidden_dims: list[int] = None,
        critic_cnn_kernel_sizes: list[int] = None,
        critic_cnn_strides: list[int] = None,
        # Actor MLP特征提取器配置
        actor_mlp_feature_dim: int = 64,
        actor_mlp_hidden_dims: tuple[int] | list[int] = [128],
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()
        
        # 高程图相关配置
        self.vision_spatial_size = vision_spatial_size
        self.vision_feature_dim = vision_feature_dim
        self.actor_mlp_feature_dim = actor_mlp_feature_dim
        self.elevation_history_length = elevation_history_length
        
        # Critic CNN配置（如果没有指定，则使用Actor的配置）
        if critic_cnn_hidden_dims is None:
            critic_cnn_hidden_dims = actor_cnn_hidden_dims
        if critic_cnn_kernel_sizes is None:
            critic_cnn_kernel_sizes = actor_cnn_kernel_sizes
        if critic_cnn_strides is None:
            critic_cnn_strides = actor_cnn_strides
        
        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == "height_scan_policy":
                # 跳过高程图，单独处理
                continue
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if obs_group == "height_scan_critic":
                continue
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        # Actor输入维度 = 本体特征 + 视觉特征
        actor_input_dim = actor_mlp_feature_dim + vision_feature_dim
        
        # Critic输入维度 = 本体特权观测 + 视觉特征(通过2DCNN提取)
        critic_input_dim = num_critic_obs + vision_feature_dim

        self.state_dependent_std = state_dependent_std

        # Actor网络：本体特征 + 视觉特征 -> MLP -> 动作
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")
        print(f"  Actor input dim: {actor_input_dim} (proprio feature: {actor_mlp_feature_dim} + vision feature: {vision_feature_dim})")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        
        # Actor MLP特征提取器：本体观测 -> MLP -> 本体特征
        self.actor_mlp_extractor = MLP(
            num_actor_obs, 
            actor_mlp_feature_dim, 
            actor_mlp_hidden_dims, 
            activation
        )
        print(f"Actor MLP Extractor: {self.actor_mlp_extractor}")
        print(f"  Extractor input dim: {num_actor_obs} -> output dim: {actor_mlp_feature_dim}")
        
        # Actor 2DCNN编码器：多帧高程图历史 -> 2DCNN -> 视觉特征
        self.elevation_2dcnn_encoder_actor = Elevation2DCNNEncoder(
            in_channels=elevation_history_length,  # 输入通道数 = 历史帧数
            hidden_dims=actor_cnn_hidden_dims,
            kernel_sizes=actor_cnn_kernel_sizes,
            strides=actor_cnn_strides,
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        print(f"Actor 2DCNN Encoder (history length={elevation_history_length}): {self.elevation_2dcnn_encoder_actor}")
        
        # Critic 2DCNN编码器：多帧高程图历史 -> 2DCNN -> 视觉特征
        self.elevation_2dcnn_encoder_critic = Elevation2DCNNEncoder(
            in_channels=elevation_history_length,  # 输入通道数 = 历史帧数
            hidden_dims=critic_cnn_hidden_dims,
            kernel_sizes=critic_cnn_kernel_sizes,
            strides=critic_cnn_strides,
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size
        )
        print(f"Critic 2DCNN Encoder (history length={elevation_history_length}): {self.elevation_2dcnn_encoder_critic}")

        # Critic网络：本体特权观测 + 视觉特征(2DCNN提取) -> MLP -> 价值
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")
        print(f"  Critic input dim: {critic_input_dim} (proprio: {num_critic_obs} + vision feature: {vision_feature_dim})")

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
        """归一化多帧高程图历史
        
        Args:
            height_map: 高程图历史，形状为 [B, T, H, W]，T为历史帧数
        
        Returns:
            归一化后的高程图历史，形状与输入相同
        """
        # 输入应该是 [B, T, H, W] 格式
        # 对每个历史帧分别归一化
        
        # 归一化：对每个历史帧计算均值
        height_map_mean = height_map.mean(dim=(-2, -1), keepdim=True)  # [B, T, 1, 1]
        height_map = torch.clip((height_map - height_map_mean) / 0.6, -3.0, 3.0)
        
        return height_map

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """训练时的动作采样
        
        处理流程：
        1. 提取多帧高程图历史 -> 归一化 -> Actor 2DCNN -> 视觉特征
        2. 提取本体观测 -> 归一化 -> MLP -> 本体特征
        3. concat本体特征和视觉特征 -> Actor网络 -> 动作
        """
        # 1. 提取高程图历史和本体观测
        height_map = obs["height_scan_policy"]  # [B, T, H, W]
        current_proprio_obs = obs["policy"]
        
        # 2. 对高程图历史进行归一化
        sampled_height_map = self._normalize_elevation_map(height_map)
        
        # 3. 应用观测归一化到本体观测
        current_proprio_obs = self.actor_obs_normalizer(current_proprio_obs)
        
        # 4. 提取高程图特征（Actor 2DCNN）
        vision_features = self.elevation_2dcnn_encoder_actor(sampled_height_map)
        
        # 5. 提取本体特征（MLP）
        proprio_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 6. 融合本体特征和视觉特征
        fused_features = torch.cat((proprio_features, vision_features), dim=-1)
        
        # 7. Actor输出动作
        self._update_distribution(fused_features)
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理时的确定性动作
        
        处理流程：
        1. 提取多帧高程图历史 -> 归一化 -> Actor 2DCNN -> 视觉特征
        2. 提取本体观测 -> 归一化 -> MLP -> 本体特征
        3. concat本体特征和视觉特征 -> Actor网络 -> 动作
        """
        # 1. 提取高程图历史和本体观测
        height_map = obs["height_scan_policy"]  # [B, T, H, W]
        current_proprio_obs = obs["policy"]
        
        # 2. 对高程图历史进行归一化
        sampled_height_map = self._normalize_elevation_map(height_map)
        
        # 3. 应用观测归一化到本体观测
        current_proprio_obs = self.actor_obs_normalizer(current_proprio_obs)
        
        # 4. 提取高程图特征（Actor 2DCNN）
        vision_features = self.elevation_2dcnn_encoder_actor(sampled_height_map)
        
        # 5. 提取本体特征（MLP）
        proprio_features = self.actor_mlp_extractor(current_proprio_obs)
        
        # 6. 融合本体特征和视觉特征
        fused_features = torch.cat((proprio_features, vision_features), dim=-1)
        
        # 7. Actor输出动作
        if self.state_dependent_std:
            return self.actor(fused_features)[..., 0, :], self.extra_info
        else:
            return self.actor(fused_features), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值
        
        处理流程：
        1. 提取多帧高程图历史 -> 归一化 -> Critic 2DCNN -> 视觉特征
        2. 提取本体特权观测 -> 归一化
        3. concat本体观测和视觉特征 -> Critic网络 -> 价值
        """
        # 1. 提取高程图历史和本体观测
        height_map = obs["height_scan_critic"]  # [B, T, H, W]
        current_proprio_obs = obs["critic"]
        
        # 2. 对高程图历史进行归一化
        sampled_height_map = self._normalize_elevation_map(height_map)
        
        # 3. 应用观测归一化到本体观测
        current_proprio_obs = self.critic_obs_normalizer(current_proprio_obs)
        
        # 4. 提取高程图特征（Critic 2DCNN）
        vision_features = self.elevation_2dcnn_encoder_critic(sampled_height_map)
        
        # 5. 融合本体观测和视觉特征
        fused_features = torch.cat((current_proprio_obs, vision_features), dim=-1)
        
        # 6. Critic输出价值
        return self.critic(fused_features)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取actor的本体观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group == "height_scan_policy":
                # 跳过高程图，单独处理
                continue
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1) if obs_list else torch.empty(obs[self.obs_groups["policy"][0]].shape[0], 0)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """获取critic的本体特权观测(排除高程图)"""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            if obs_group == "height_scan_critic":
                # 跳过高程图，单独处理
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
        """Load parameters of actor-critic model.

        Args:
            state_dict: State dictionary of model.
            strict: Whether to strictly enforce that keys in `state_dict` match the keys returned by this module's
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
    """Mode13A5的ONNX导出器"""
    
    def __init__(self, policy: ActorCriticMode13A5, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # 复制策略参数
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        if hasattr(policy, "elevation_2dcnn_encoder_actor"):
            self.elevation_encoder = copy.deepcopy(policy.elevation_2dcnn_encoder_actor)
        if hasattr(policy, "actor_mlp_extractor"):
            self.actor_mlp_extractor = copy.deepcopy(policy.actor_mlp_extractor)
        
        # 从actor获取相关维度
        self.vision_feature_dim = policy.vision_feature_dim
        self.actor_mlp_feature_dim = policy.actor_mlp_feature_dim
        self.proprio_obs_dim = policy.actor_mlp_extractor[0].in_features
        self.vision_spatial_size = policy.vision_spatial_size
        self.elevation_history_length = policy.elevation_history_length

        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
    
    def forward(self, obs_concat):
        """
        Args:
            obs_concat: 拼接的观测，形状为 [batch_size, total_dim]
                       前面是本体观测，后面是高程图历史
                       total_dim = proprio_obs_dim + history_length * height * width
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_concat.shape[0]
        
        # 计算高程图展平后的维度
        history_length = self.elevation_history_length
        height, width = self.vision_spatial_size
        elevation_flat_dim = history_length * height * width
        
        # 从输入中切片分出本体观测和高程图
        proprio_obs = obs_concat[:, :self.proprio_obs_dim]  # [batch_size, proprio_obs_dim]
        elevation_flat = obs_concat[:, self.proprio_obs_dim:]  # [batch_size, history_length * height * width]
        
        # 将高程图重塑为 [batch_size, history_length, height, width]
        elevation_obs = elevation_flat.view(batch_size, history_length, height, width)
        
        # 对本体观测进行归一化
        proprio_obs = self.normalizer(proprio_obs)
        
        # 提取本体特征（MLP）
        proprio_features = self.actor_mlp_extractor(proprio_obs)
        
        # 对高程图历史进行2DCNN编码（输入已经是多通道格式）
        vision_features = self.elevation_encoder(elevation_obs)
        
        # 融合本体特征和视觉特征
        fused_features = torch.cat([proprio_features, vision_features], dim=-1)
        
        # 输出动作
        actions_mean = self.actor(fused_features)
        return actions_mean
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        
        history_length = self.elevation_history_length
        height, width = self.vision_spatial_size
        total_dim = self.proprio_obs_dim + history_length * height * width
        obs_concat = torch.zeros(1, total_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (Mode13A5):")
        print(f"{'='*80}")
        print(f"  本体观测维度:       {self.proprio_obs_dim}")
        print(f"  本体特征维度:       {self.actor_mlp_feature_dim}")
        print(f"  视觉特征维度:       {self.vision_feature_dim}")
        print(f"  高程图历史长度:     {history_length}")
        print(f"  高程图空间维度:     ({height}, {width})")
        print(f"  总输入维度:         {total_dim}")
        print(f"  输入: 拼接观测,     shape: [batch, {total_dim}]")
        print(f"    - 前{self.proprio_obs_dim}维: 本体观测")
        print(f"    - 后{history_length * height * width}维: 高程图历史(展平)")
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
