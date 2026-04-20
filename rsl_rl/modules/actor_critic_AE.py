# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticAE(nn.Module):
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
        encoder_hidden_dims: tuple[int] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int] | list[int] = (256, 256),
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        num_history_len: int = 5,
        num_latent: int = 16,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic_AE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if num_latent < 0:
            raise ValueError(f"num_latent must be non-negative. Received: {num_latent}.")

        # 传递回Env的额外信息
        self.extra_info = dict()

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.num_history_len = num_history_len
        self.num_latent = num_latent

        # Observation dimensions for the separate AE data paths.
        num_actor_obs = self._sum_obs_dims(obs, obs_groups["policy"])
        num_history_obs = self._sum_obs_dims(obs, obs_groups["policy_history"])
        num_critic_obs = self._sum_obs_dims(obs, obs_groups["critic"])
        num_critic_vel_obs = self._sum_obs_dims(obs, obs_groups["critic_vel"]) if "critic_vel" in obs_groups else 3
        num_critic_prop_obs = (
            self._sum_obs_dims(obs, obs_groups["critic_prop"]) if "critic_prop" in obs_groups else 0
        )

        if num_critic_vel_obs < 3:
            raise ValueError(f"critic_vel observation must contain at least 3 values. Received: {num_critic_vel_obs}.")
        if self.num_latent > 0 and num_critic_prop_obs <= 0:
            raise ValueError("critic_prop observation group is required when num_latent > 0.")

        self.num_actor_obs = num_actor_obs
        self.num_history_obs = num_history_obs
        self.num_critic_obs = num_critic_obs
        self.num_critic_prop_obs = num_critic_prop_obs


        # Actor
        actor_input_dim = num_actor_obs + 3 + self.num_latent
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor/current proprioception normalization.
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
            self.history_obs_normalizer = EmpiricalNormalization(num_history_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            self.history_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Encoder: proprioceptive history -> [base linear velocity, optional latent].
        self.encoder = MLP(num_history_obs, 3 + self.num_latent, encoder_hidden_dims, activation)
        print(f"Encoder MLP: {self.encoder}")

        # Decoder: latent -> next critic_prop. Disabled when latent dimension is 0.
        if self.num_latent > 0:
            self.decoder = MLP(self.num_latent, num_critic_prop_obs, decoder_hidden_dims, activation)
            print(f"Decoder MLP: {self.decoder}")
        else:
            self.decoder = None
            print("Decoder MLP: disabled because num_latent=0")

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

    @staticmethod
    def _sum_obs_dims(obs: TensorDict, obs_group_names: list[str]) -> int:
        num_obs = 0
        for obs_group in obs_group_names:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_obs += obs[obs_group].shape[-1]
        return num_obs

    def _update_distribution(self, obs: torch.Tensor) -> None:
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

    def encoder_forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the AE encoder and split velocity from the optional latent."""
        code = self.encoder(obs_history)
        est_vel = code[:, :3]
        latent = code[:, 3:]
        return est_vel, latent

    def decoder_forward(self, latent: torch.Tensor) -> torch.Tensor | None:
        if self.decoder is None:
            return None
        return self.decoder(latent)

    def _make_actor_input(
        self, actor_obs: torch.Tensor, est_vel: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        if self.num_latent > 0:
            return torch.cat((est_vel, latent, actor_obs), dim=-1)
        return torch.cat((est_vel, actor_obs), dim=-1)

    def _get_normalized_actor_inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        history_obs = self.history_obs_normalizer(self.get_history_obs(obs))
        return actor_obs, history_obs

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs, history_obs = self._get_normalized_actor_inputs(obs)
        est_vel, latent = self.encoder_forward(history_obs)
        observation = self._make_actor_input(actor_obs, est_vel.detach(), latent.detach())
        self._update_distribution(observation)

        self.extra_info = {"est_vel": est_vel.detach()}
        if self.num_latent > 0:
            self.extra_info["latent"] = latent.detach()
            critic_prop_predict = self.decoder_forward(latent)
            if critic_prop_predict is not None:
                self.extra_info["critic_prop_predict"] = critic_prop_predict.detach()
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs, history_obs = self._get_normalized_actor_inputs(obs)
        est_vel, latent = self.encoder_forward(history_obs)
        observation = self._make_actor_input(actor_obs, est_vel.detach(), latent.detach())

        self.extra_info = {"est_vel": est_vel.detach()}
        if self.num_latent > 0:
            self.extra_info["latent"] = latent.detach()
            critic_prop_predict = self.decoder_forward(latent)
            if critic_prop_predict is not None:
                self.extra_info["critic_prop_predict"] = critic_prop_predict.detach()

        if self.state_dependent_std:
            return self.actor(observation)[..., 0, :], self.extra_info
        else:
            return self.actor(observation), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        history_groups = self.obs_groups.get("policy_history", self.obs_groups["policy"])
        obs_list = [obs[obs_group] for obs_group in history_groups]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_vel_obs(self, obs: TensorDict) -> torch.Tensor:
        if "critic_vel" in self.obs_groups:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic_vel"]]
            return torch.cat(obs_list, dim=-1)
        return self.get_critic_obs(obs)[:, :3]

    def get_critic_prop_obs(self, obs: TensorDict) -> torch.Tensor:
        if "critic_prop" not in self.obs_groups:
            raise ValueError("critic_prop observation group is required for AE decoder training.")
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic_prop"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
            self.history_obs_normalizer.update(self.get_history_obs(obs))
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

    def update_encoder(
        self,
        obs_batch: TensorDict,
        next_observations_batch: TensorDict,
        encoder_optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
    ) -> dict[str, float]:
        """Update the encoder and optional decoder from rollout observations."""
        history_obs = self.history_obs_normalizer(self.get_history_obs(obs_batch))
        est_vel, latent = self.encoder_forward(history_obs)

        vel_target = self.get_critic_vel_obs(obs_batch)[:, :3].detach()
        vel_loss = nn.MSELoss()(est_vel, vel_target)
        total_loss = vel_loss

        loss_dict = {
            "vel_loss": vel_loss.item(),
        }

        if self.num_latent > 0:
            critic_prop_predict = self.decoder_forward(latent)
            critic_prop_target = self.get_critic_prop_obs(next_observations_batch).detach()
            prop_loss = nn.MSELoss()(critic_prop_predict, critic_prop_target)
            total_loss = total_loss + prop_loss
            loss_dict["prop_loss"] = prop_loss.item()

        encoder_optimizer.zero_grad()
        total_loss.backward()

        encoder_params = [p for group in encoder_optimizer.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(encoder_params, max_grad_norm)

        encoder_optimizer.step()
        encoder_optimizer.zero_grad(set_to_none=True)

        loss_dict["total_loss"] = total_loss.item()
        return loss_dict

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器"""
        import torch.optim as optim

        optimizer_params = [
            {"params": self.actor.parameters()},
            {"params": self.critic.parameters()},
        ]
        if not self.state_dependent_std:
            optimizer_params.append(
                {"params": [self.std] if self.noise_std_type == "scalar" else [self.log_std]}
            )
        optimizer = optim.Adam(optimizer_params, lr=learning_rate)

        encoder_params = [{"params": self.encoder.parameters()}]
        if self.decoder is not None:
            encoder_params.append({"params": self.decoder.parameters()})
        encoder_optimizer = optim.Adam(encoder_params, lr=learning_rate)

        return {
            "optimizer": optimizer,
            "encoder_optimizer": encoder_optimizer,
        }

    def export_to_onnx(
        self,
        path: str,
        filename: str = "AE_policy.onnx",
        normalizer: torch.nn.Module | None = None,
        verbose: bool = False,
    ) -> None:
        """将AE策略导出为ONNX格式"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        exporter = _AEOnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _AEOnnxPolicyExporter(torch.nn.Module):
    """AE策略的ONNX导出器。

    The ONNX input is a concatenation of [policy_new, policy_history].
    """

    def __init__(self, policy: ActorCriticAE, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.state_dependent_std = policy.state_dependent_std
        self.num_actor_obs = policy.num_actor_obs
        self.num_history_obs = policy.num_history_obs
        self.num_latent = policy.num_latent

        if normalizer is not None:
            self.actor_obs_normalizer = copy.deepcopy(normalizer)
        else:
            self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.history_obs_normalizer = copy.deepcopy(policy.history_obs_normalizer)

    def forward(self, x: torch.Tensor):
        actor_obs = x[:, : self.num_actor_obs]
        history_obs = x[:, self.num_actor_obs : self.num_actor_obs + self.num_history_obs]

        actor_obs = self.actor_obs_normalizer(actor_obs)
        history_obs = self.history_obs_normalizer(history_obs)

        code = self.encoder(history_obs)
        est_vel = code[:, :3]
        latent = code[:, 3:]

        if self.num_latent > 0:
            actor_input = torch.cat((est_vel.detach(), latent.detach(), actor_obs), dim=-1)
        else:
            actor_input = torch.cat((est_vel.detach(), actor_obs), dim=-1)

        actions_mean = self.actor(actor_input)
        if self.state_dependent_std:
            actions_mean = actions_mean[..., 0, :]

        if self.num_latent > 0:
            return actions_mean, est_vel, latent
        return actions_mean, est_vel

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        obs = torch.zeros(1, self.num_actor_obs + self.num_history_obs)
        output_names = ["actions", "est_vel"]
        if self.num_latent > 0:
            output_names.append("latent")
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=output_names,
            dynamic_axes={},
        )
