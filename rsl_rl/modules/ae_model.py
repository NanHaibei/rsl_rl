# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.modules.rnn import HiddenState
from rsl_rl.utils import resolve_class, unpad_trajectories


def _sum_obs_dims(obs: TensorDict, obs_group_names: list[str]) -> tuple[list[str], int]:
    """Validate 1D observation groups and return group list and summed feature dimension."""
    obs_dim = 0
    for obs_group in obs_group_names:
        if len(obs[obs_group].shape) != 2:
            raise ValueError(
                f"Estimator models expect 1D critic observations, got shape {obs[obs_group].shape} for '{obs_group}'."
            )
        obs_dim += obs[obs_group].shape[-1]
    return list(obs_group_names), obs_dim


def _resolve_actor_history_obs(
    obs: TensorDict,
    obs_group_names: list[str],
    history_length: int | None,
) -> tuple[list[str], int, int, int, bool]:
    """Validate that actor observations include history frames and infer dimensions."""
    if not obs_group_names:
        raise ValueError("Actor obs_groups must contain at least one observation group.")

    ndims = {obs[group].ndim for group in obs_group_names}
    if ndims == {3}:
        hist_lens = {int(obs[group].shape[1]) for group in obs_group_names}
        if len(hist_lens) != 1:
            raise ValueError(f"All 3D actor observation groups must share the same history length, got {hist_lens}.")
        inferred_history_len = next(iter(hist_lens))
        if inferred_history_len <= 1:
            raise ValueError(
                f"Actor observation must contain history frames (history_length > 1), got shape "
                f"{tuple(obs[obs_group_names[0]].shape)}."
            )
        if history_length is not None and history_length != inferred_history_len:
            raise ValueError(
                f"Configured history_length={history_length} does not match actor observation history dimension "
                f"{inferred_history_len}."
            )
        single_obs_dim = sum(int(obs[group].shape[2]) for group in obs_group_names)
        obs_dim = inferred_history_len * single_obs_dim
        return list(obs_group_names), inferred_history_len, single_obs_dim, obs_dim, True

    if ndims == {2}:
        if history_length is None or history_length <= 1:
            raise ValueError(
                "Actor observation has 2D shape with no detectable history frames. Configure env_cfg with "
                "history_length > 1 and flatten_history_dim=False (recommended), or use history_order='time' "
                "and pass history_length > 1."
            )
        single_obs_dim = 0
        for group in obs_group_names:
            dim = int(obs[group].shape[-1])
            if dim % history_length != 0:
                raise ValueError(
                    f"2D actor observation '{group}' dimension ({dim}) is not divisible by history_length={history_length}."
                )
            single_obs_dim += dim // history_length
        obs_dim = history_length * single_obs_dim
        return list(obs_group_names), history_length, single_obs_dim, obs_dim, False

    raise ValueError(
        f"Actor observations must be uniformly 3D (B, H, D) or 2D (B, H*D), got ndims={sorted(ndims)}."
    )


class _EstimatorActorBase(nn.Module):
    """Base actor model with history encoder and optional next-state decoder."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        num_latent: int = 16,
        history_length: int | None = None,
        vel_loss_coef: float = 1.0,
        prop_loss_coef: float = 1.0,
        encoder_learning_rate: float | None = None,
        encoder_follow_policy_lr: bool = True,
        encoder_max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__()

        if num_latent < 0:
            raise ValueError(f"num_latent must be non-negative. Received: {num_latent}.")
        if len(encoder_hidden_dims) == 0:
            raise ValueError("encoder_hidden_dims must contain at least one hidden dimension.")
        if num_latent > 0 and len(decoder_hidden_dims) == 0:
            raise ValueError("decoder_hidden_dims must contain at least one hidden dimension when num_latent > 0.")
        if vel_loss_coef < 0.0:
            raise ValueError(f"vel_loss_coef must be non-negative. Received: {vel_loss_coef}.")
        if prop_loss_coef < 0.0:
            raise ValueError(f"prop_loss_coef must be non-negative. Received: {prop_loss_coef}.")
        if encoder_max_grad_norm <= 0.0:
            raise ValueError(f"encoder_max_grad_norm must be positive. Received: {encoder_max_grad_norm}.")

        (
            self.obs_groups,
            self.history_length,
            self.single_obs_dim,
            self.obs_dim,
            self._actor_obs_is_3d,
        ) = _resolve_actor_history_obs(obs, obs_groups[obs_set], history_length)

        if "critic_vel" in obs_groups:
            self.critic_vel_obs_groups, self.critic_vel_obs_dim = _sum_obs_dims(obs, obs_groups["critic_vel"])
        elif "critic" in obs_groups:
            self.critic_vel_obs_groups, self.critic_vel_obs_dim = _sum_obs_dims(obs, obs_groups["critic"])
        else:
            raise ValueError("obs_groups must contain 'critic_vel' or 'critic' for velocity supervision.")
        if self.critic_vel_obs_dim < 3:
            raise ValueError(f"critic_vel observation must contain at least 3 values. Got: {self.critic_vel_obs_dim}.")

        if "critic_prop" in obs_groups:
            self.critic_prop_obs_groups, self.critic_prop_obs_dim = _sum_obs_dims(obs, obs_groups["critic_prop"])
        else:
            self.critic_prop_obs_groups, self.critic_prop_obs_dim = [], 0
        if num_latent > 0 and self.critic_prop_obs_dim <= 0:
            raise ValueError("obs_groups['critic_prop'] is required when num_latent > 0.")

        self.num_latent = num_latent
        self.vel_loss_coef = float(vel_loss_coef)
        self.prop_loss_coef = float(prop_loss_coef)
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_follow_policy_lr = encoder_follow_policy_lr
        self.encoder_max_grad_norm = float(encoder_max_grad_norm)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_class, dist_cfg = resolve_class(distribution_cfg)
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        actor_input_dim = self.single_obs_dim + 3 + self.num_latent
        self.mlp = MLP(actor_input_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        self.decoder: MLP | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Compute deterministic or stochastic actions matching the MLPModel contract."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization and self.obs_groups:
            self.obs_normalizer.update(self.get_actor_obs(obs))  # type: ignore

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return time-major flattened history observations of shape (B, H * D)."""
        if self._actor_obs_is_3d:
            stacked = (
                obs[self.obs_groups[0]]
                if len(self.obs_groups) == 1
                else torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            )
            return stacked.reshape(stacked.shape[0], self.obs_dim)
        if len(self.obs_groups) == 1:
            return obs[self.obs_groups[0]]
        batch_size = obs[self.obs_groups[0]].shape[0]
        reshaped = [obs[group].reshape(batch_size, self.history_length, -1) for group in self.obs_groups]
        return torch.cat(reshaped, dim=-1).reshape(batch_size, self.obs_dim)

    def get_critic_vel_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.critic_vel_obs_groups], dim=-1)

    def get_decoder_target_obs(self, obs: TensorDict) -> torch.Tensor:
        if not self.critic_prop_obs_groups:
            raise ValueError("critic_prop observation group is required when decoding next-step proprioception.")
        return torch.cat([obs[group] for group in self.critic_prop_obs_groups], dim=-1)

    @property
    def requires_next_decoder_obs(self) -> bool:
        return self.num_latent > 0

    @property
    def decoder_target_dim(self) -> int:
        return self.critic_prop_obs_dim if self.num_latent > 0 else 0

    def policy_parameters(self) -> list[nn.Parameter]:
        """Return only policy MLP and action distribution parameters updated by PPO."""
        params = list(self.mlp.parameters())
        if self.distribution is not None:
            params.extend(self.distribution.parameters())
        return params

    def encoder_parameters(self) -> list[nn.Parameter]:
        raise NotImplementedError

    def compute_encoder_loss(
        self,
        obs_batch: TensorDict,
        next_decoder_target: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def _compute_prop_loss(
        self,
        latent_for_decoder: torch.Tensor,
        next_decoder_target: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder is disabled when num_latent == 0.")
        if next_decoder_target is None:
            raise ValueError("next_decoder_target is required when num_latent > 0.")
        pred = self.decoder(latent_for_decoder)
        target = next_decoder_target.detach()
        if valid_mask is not None:
            mask = valid_mask.bool()
            if mask.any():
                return F.mse_loss(pred[mask], target[mask])
            return pred.sum() * 0.0
        return F.mse_loss(pred, target)


class AEModel(_EstimatorActorBase):
    """Autoencoder velocity/context estimator actor compatible with RSL-RL 3.0 PPO."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        num_latent: int = 16,
        history_length: int | None = None,
        vel_loss_coef: float = 1.0,
        prop_loss_coef: float = 1.0,
        encoder_learning_rate: float | None = None,
        encoder_follow_policy_lr: bool = True,
        encoder_max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            num_latent=num_latent,
            history_length=history_length,
            vel_loss_coef=vel_loss_coef,
            prop_loss_coef=prop_loss_coef,
            encoder_learning_rate=encoder_learning_rate,
            encoder_follow_policy_lr=encoder_follow_policy_lr,
            encoder_max_grad_norm=encoder_max_grad_norm,
        )

        self.encoder = MLP(self.obs_dim, 3 + self.num_latent, encoder_hidden_dims, activation)
        if self.num_latent > 0:
            self.decoder = MLP(self.num_latent, self.critic_prop_obs_dim, decoder_hidden_dims, activation)
        else:
            self.decoder = None

    def encoder_forward(self, history_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(history_obs)
        return code[:, :3], code[:, 3:]

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        history_obs = self.obs_normalizer(self.get_actor_obs(obs))
        current_obs = history_obs[:, -self.single_obs_dim :]
        est_vel, latent = self.encoder_forward(history_obs)
        if self.num_latent > 0:
            return torch.cat((est_vel.detach(), latent.detach(), current_obs), dim=-1)
        return torch.cat((est_vel.detach(), current_obs), dim=-1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        params = list(self.encoder.parameters())
        if self.decoder is not None:
            params.extend(self.decoder.parameters())
        return params

    def compute_encoder_loss(
        self,
        obs_batch: TensorDict,
        next_decoder_target: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        history_obs = self.obs_normalizer(self.get_actor_obs(obs_batch))
        est_vel, latent = self.encoder_forward(history_obs)

        vel_target = self.get_critic_vel_obs(obs_batch)[:, :3].detach()
        vel_loss = F.mse_loss(est_vel, vel_target)
        total_loss = self.vel_loss_coef * vel_loss
        loss_dict = {"vel_loss": vel_loss.item()}

        if self.num_latent > 0:
            prop_loss = self._compute_prop_loss(latent, next_decoder_target, valid_mask)
            total_loss = total_loss + self.prop_loss_coef * prop_loss
            loss_dict["prop_loss"] = prop_loss.item()

        loss_dict["encoder_total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def as_jit(self) -> nn.Module:
        if self.num_latent > 0:
            return _TorchAELatentModel(self)
        return _TorchAEVelModel(self)

    def as_onnx(self) -> nn.Module:
        if self.num_latent > 0:
            return _OnnxAELatentModel(self)
        return _OnnxAEVelModel(self)


class _AEExportBase(nn.Module):
    """Shared single-input flattened history export core for AEModel."""

    is_recurrent: bool = False

    def __init__(self, model: AEModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.single_obs_dim = model.single_obs_dim
        self.input_size = model.obs_dim

    def _forward_core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_obs = self.obs_normalizer(x)
        current_obs = history_obs[:, -self.single_obs_dim :]
        code = self.encoder(history_obs)
        est_vel = code[:, :3]
        latent = code[:, 3:]
        return current_obs, est_vel, latent


class _TorchAELatentModel(_AEExportBase):
    """TorchScript export wrapper for AEModel with num_latent > 0."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_obs, est_vel, latent = self._forward_core(x)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, latent, current_obs), dim=-1)))
        return actions, est_vel, latent

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchAEVelModel(_AEExportBase):
    """TorchScript export wrapper for pure velocity estimation AEModel (num_latent == 0)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_obs, est_vel, _ = self._forward_core(x)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, current_obs), dim=-1)))
        return actions, est_vel

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxAELatentModel(_AEExportBase):
    """ONNX export wrapper for AEModel with num_latent > 0."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_obs, est_vel, latent = self._forward_core(x)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, latent, current_obs), dim=-1)))
        return actions, est_vel, latent

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "est_vel", "latent"]


class _OnnxAEVelModel(_AEExportBase):
    """ONNX export wrapper for pure velocity estimation AEModel (num_latent == 0)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_obs, est_vel, _ = self._forward_core(x)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, current_obs), dim=-1)))
        return actions, est_vel

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "est_vel"]
