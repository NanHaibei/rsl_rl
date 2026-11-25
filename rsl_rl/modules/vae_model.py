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

from rsl_rl.modules.ae_model import _EstimatorActorBase
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.rnn import HiddenState
from rsl_rl.utils import resolve_nn_activation


class VAEModel(_EstimatorActorBase):
    """DWAQ variational autoencoder velocity/context estimator actor compatible with RSL-RL 3.0 PPO."""

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
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        num_latent: int = 16,
        history_length: int | None = None,
        vae_beta: float = 1.0,
        vel_loss_coef: float = 1.0,
        prop_loss_coef: float = 1.0,
        encoder_learning_rate: float | None = None,
        encoder_follow_policy_lr: bool = True,
        encoder_max_grad_norm: float = 1.0,
    ) -> None:
        if vae_beta < 0.0:
            raise ValueError(f"vae_beta must be non-negative. Received: {vae_beta}.")

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

        self.vae_beta = float(vae_beta)
        encoder_feature_dim = encoder_hidden_dims[-1]
        if len(encoder_hidden_dims) == 1:
            self.encoder_backbone: nn.Module = nn.Sequential(
                nn.Linear(self.obs_dim, encoder_feature_dim),
                resolve_nn_activation(activation),
            )
        else:
            self.encoder_backbone = MLP(
                self.obs_dim,
                encoder_feature_dim,
                encoder_hidden_dims[:-1],
                activation,
                last_activation=activation,
            )
        self.encoder_vel_mean = nn.Linear(encoder_feature_dim, 3)

        if self.num_latent > 0:
            self.encoder_latent_mean: nn.Linear | None = nn.Linear(encoder_feature_dim, self.num_latent)
            self.encoder_latent_logvar: nn.Linear | None = nn.Linear(encoder_feature_dim, self.num_latent)
            self.decoder = MLP(self.num_latent, self.critic_prop_obs_dim, decoder_hidden_dims, activation)
        else:
            self.encoder_latent_mean = None
            self.encoder_latent_logvar = None
            self.decoder = None

    @staticmethod
    def reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mean, exp(logvar)) using the reparameterization trick."""
        std = torch.exp(logvar * 0.5)
        return mean + std * torch.randn_like(std)

    def encoder_forward(
        self, history_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the VAE encoder and return (est_vel, latent_mean, latent_logvar, latent_sample)."""
        x = self.encoder_backbone(history_obs)
        est_vel = self.encoder_vel_mean(x)

        if self.num_latent > 0 and self.encoder_latent_mean is not None and self.encoder_latent_logvar is not None:
            latent_mean = self.encoder_latent_mean(x)
            latent_logvar = torch.clamp(self.encoder_latent_logvar(x), min=-10.0, max=10.0)
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_mean = est_vel.new_empty(est_vel.shape[0], 0)
            latent_logvar = est_vel.new_empty(est_vel.shape[0], 0)
            latent_sample = est_vel.new_empty(est_vel.shape[0], 0)

        return est_vel, latent_mean, latent_logvar, latent_sample

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        history_obs = self.obs_normalizer(self.get_actor_obs(obs))
        current_obs = history_obs[:, -self.single_obs_dim :]
        est_vel, latent_mean, _, _ = self.encoder_forward(history_obs)
        if self.num_latent > 0:
            return torch.cat((est_vel.detach(), latent_mean.detach(), current_obs), dim=-1)
        return torch.cat((est_vel.detach(), current_obs), dim=-1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        params = list(self.encoder_backbone.parameters()) + list(self.encoder_vel_mean.parameters())
        if self.num_latent > 0:
            if self.encoder_latent_mean is not None:
                params.extend(self.encoder_latent_mean.parameters())
            if self.encoder_latent_logvar is not None:
                params.extend(self.encoder_latent_logvar.parameters())
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
        est_vel, latent_mean, latent_logvar, latent_sample = self.encoder_forward(history_obs)

        vel_target = self.get_critic_vel_obs(obs_batch)[:, :3].detach()
        vel_loss = F.mse_loss(est_vel, vel_target)
        total_loss = self.vel_loss_coef * vel_loss
        loss_dict = {"vel_loss": vel_loss.item()}

        if self.num_latent > 0:
            prop_loss = self._compute_prop_loss(latent_sample, next_decoder_target, valid_mask)
            kl_loss = -0.5 * torch.mean(
                torch.sum(1.0 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1)
            )
            total_loss = total_loss + self.prop_loss_coef * prop_loss + self.vae_beta * kl_loss
            loss_dict["prop_loss"] = prop_loss.item()
            loss_dict["kl_loss"] = kl_loss.item()

        loss_dict["encoder_total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def as_jit(self) -> nn.Module:
        if self.num_latent > 0:
            return _TorchVAELatentModel(self)
        return _TorchVAEVelModel(self)

    def as_onnx(self) -> nn.Module:
        if self.num_latent > 0:
            return _OnnxVAELatentModel(self)
        return _OnnxVAEVelModel(self)


class _VAEExportBase(nn.Module):
    """Shared single-input flattened history export core for VAEModel."""

    is_recurrent: bool = False

    def __init__(self, model: VAEModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_backbone = copy.deepcopy(model.encoder_backbone)
        self.encoder_vel_mean = copy.deepcopy(model.encoder_vel_mean)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.single_obs_dim = model.single_obs_dim
        self.input_size = model.obs_dim

    def _encode_base(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_obs = self.obs_normalizer(x)
        current_obs = history_obs[:, -self.single_obs_dim :]
        features = self.encoder_backbone(history_obs)
        est_vel = self.encoder_vel_mean(features)
        return current_obs, features, est_vel


class _VAEExportLatentBase(_VAEExportBase):
    """Shared latent-enabled export base for VAEModel."""

    def __init__(self, model: VAEModel) -> None:
        super().__init__(model)
        if model.encoder_latent_mean is None:
            raise ValueError("encoder_latent_mean is required for latent VAE export.")
        self.encoder_latent_mean = copy.deepcopy(model.encoder_latent_mean)

    def _forward_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_obs, features, est_vel = self._encode_base(x)
        latent_mean = self.encoder_latent_mean(features)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, latent_mean, current_obs), dim=-1)))
        return actions, est_vel, latent_mean


class _VAEExportVelBase(_VAEExportBase):
    """Shared pure-velocity export base for VAEModel (num_latent == 0)."""

    def _forward_vel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_obs, _, est_vel = self._encode_base(x)
        actions = self.deterministic_output(self.mlp(torch.cat((est_vel, current_obs), dim=-1)))
        return actions, est_vel


class _TorchVAELatentModel(_VAEExportLatentBase):
    """TorchScript export wrapper for VAEModel with num_latent > 0."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_latent(x)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchVAEVelModel(_VAEExportVelBase):
    """TorchScript export wrapper for pure velocity estimation VAEModel (num_latent == 0)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_vel(x)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxVAELatentModel(_VAEExportLatentBase):
    """ONNX export wrapper for VAEModel with num_latent > 0."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_latent(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "est_vel", "latent_mean"]


class _OnnxVAEVelModel(_VAEExportVelBase):
    """ONNX export wrapper for pure velocity estimation VAEModel (num_latent == 0)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_vel(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "est_vel"]
