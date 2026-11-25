# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AEModel and VAEModel estimator actors."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.modules import AEModel, VAEModel
from rsl_rl.utils import resolve_callable

NUM_ENVS = 6
ACTOR_OBS_DIM = 10
HISTORY_LEN = 4
CRITIC_VEL_DIM = 3
CRITIC_PROP_DIM = 7
NUM_ACTIONS = 4


def _make_estimator_obs(
    num_envs: int = NUM_ENVS,
    history_len: int = HISTORY_LEN,
    shift: float = 0.0,
    flatten_policy: bool = False,
) -> TensorDict:
    policy = torch.randn(num_envs, history_len, ACTOR_OBS_DIM) + shift
    if flatten_policy:
        policy = policy.reshape(num_envs, history_len * ACTOR_OBS_DIM)
    return TensorDict(
        {
            "policy": policy,
            "critic_vel": torch.randn(num_envs, CRITIC_VEL_DIM) + shift,
            "critic_prop": torch.randn(num_envs, CRITIC_PROP_DIM) + shift,
            "critic": torch.randn(num_envs, CRITIC_VEL_DIM + CRITIC_PROP_DIM + 5) + shift,
        },
        batch_size=[num_envs],
    )


def _make_obs_groups(include_prop: bool = True) -> dict[str, list[str]]:
    groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "critic_vel": ["critic_vel"],
    }
    if include_prop:
        groups["critic_prop"] = ["critic_prop"]
    return groups


def _make_dist_cfg(dist_class: str = "GaussianDistribution") -> dict:
    return {"class_name": dist_class, "init_std": 1.0, "std_type": "scalar"}


@pytest.mark.parametrize("model_cls", [AEModel, VAEModel])
@pytest.mark.parametrize("num_latent", [0, 16])
@pytest.mark.parametrize("obs_normalization", [False, True])
@pytest.mark.parametrize("dist_class", ["GaussianDistribution", "HeteroscedasticGaussianDistribution"])
def test_estimator_forward_normalization_and_modes(
    model_cls: type[AEModel] | type[VAEModel],
    num_latent: int,
    obs_normalization: bool,
    dist_class: str,
) -> None:
    obs = _make_estimator_obs()
    obs_groups = _make_obs_groups(include_prop=num_latent > 0)
    model = model_cls(
        obs=obs,
        obs_groups=obs_groups,
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[32, 16],
        encoder_hidden_dims=[32, 16],
        decoder_hidden_dims=[16, 32],
        num_latent=num_latent,
        obs_normalization=obs_normalization,
        distribution_cfg=_make_dist_cfg(dist_class),
    )

    assert not hasattr(model, "extra_info")
    assert model.history_length == HISTORY_LEN
    assert model.single_obs_dim == ACTOR_OBS_DIM
    assert model.obs_dim == HISTORY_LEN * ACTOR_OBS_DIM
    assert model.requires_next_decoder_obs is (num_latent > 0)
    assert model.decoder_target_dim == (CRITIC_PROP_DIM if num_latent > 0 else 0)

    if obs_normalization:
        model.train()
        shifted = _make_estimator_obs(shift=10.0)
        for _ in range(10):
            model.update_normalization(shifted)

    model.eval()
    det = model(obs)
    assert det.shape == (NUM_ENVS, NUM_ACTIONS)

    torch.manual_seed(0)
    sample = model(obs, stochastic_output=True)
    assert sample.shape == (NUM_ENVS, NUM_ACTIONS)
    assert torch.allclose(det, model.output_mean, atol=1e-6)
    assert not torch.allclose(sample, det, atol=1e-6)
    assert model.get_output_log_prob(sample).shape == (NUM_ENVS,)

    if num_latent == 0:
        assert model.decoder is None
    else:
        assert model.decoder is not None


@pytest.mark.parametrize("model_cls", [AEModel, VAEModel])
def test_history_validation_and_2d_time_major_mode(model_cls: type[AEModel] | type[VAEModel]) -> None:
    # 2D policy obs without history_length must be rejected.
    obs_2d_no_hist = _make_estimator_obs(history_len=1, flatten_policy=True)
    with pytest.raises(ValueError, match="history"):
        model_cls(
            obs=obs_2d_no_hist,
            obs_groups=_make_obs_groups(include_prop=False),
            obs_set="actor",
            output_dim=NUM_ACTIONS,
            num_latent=0,
            distribution_cfg=_make_dist_cfg(),
        )

    # 3D policy obs with history_length == 1 must also be rejected.
    obs_3d_len1 = _make_estimator_obs(history_len=1, flatten_policy=False)
    with pytest.raises(ValueError, match="history"):
        model_cls(
            obs=obs_3d_len1,
            obs_groups=_make_obs_groups(include_prop=False),
            obs_set="actor",
            output_dim=NUM_ACTIONS,
            num_latent=0,
            distribution_cfg=_make_dist_cfg(),
        )

    # 2D flattened time-major obs is accepted when history_length > 1 is explicitly passed.
    obs_2d_hist = _make_estimator_obs(history_len=HISTORY_LEN, flatten_policy=True)
    model = model_cls(
        obs=obs_2d_hist,
        obs_groups=_make_obs_groups(include_prop=False),
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        num_latent=0,
        history_length=HISTORY_LEN,
        distribution_cfg=_make_dist_cfg(),
    )
    assert model.history_length == HISTORY_LEN
    assert model.single_obs_dim == ACTOR_OBS_DIM
    assert model(obs_2d_hist).shape == (NUM_ENVS, NUM_ACTIONS)

    # Non-divisible 2D dimension is rejected.
    with pytest.raises(ValueError, match="divisible"):
        model_cls(
            obs=obs_2d_hist,
            obs_groups=_make_obs_groups(include_prop=False),
            obs_set="actor",
            output_dim=NUM_ACTIONS,
            num_latent=0,
            history_length=3,
            distribution_cfg=_make_dist_cfg(),
        )


def test_vae_single_encoder_hidden_dim_and_resolve_callable() -> None:
    assert resolve_callable("AEModel") is AEModel
    assert resolve_callable("VAEModel") is VAEModel

    obs = _make_estimator_obs()
    model = VAEModel(
        obs=obs,
        obs_groups=_make_obs_groups(include_prop=True),
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[16],
        encoder_hidden_dims=[24],
        decoder_hidden_dims=[16],
        num_latent=8,
        distribution_cfg=_make_dist_cfg(),
    )
    out = model(obs)
    assert out.shape == (NUM_ENVS, NUM_ACTIONS)


@pytest.mark.parametrize("model_cls", [AEModel, VAEModel])
@pytest.mark.parametrize("num_latent", [0, 8])
def test_gradient_isolation_between_policy_and_encoder(
    model_cls: type[AEModel] | type[VAEModel], num_latent: int
) -> None:
    obs = _make_estimator_obs()
    model = model_cls(
        obs=obs,
        obs_groups=_make_obs_groups(include_prop=num_latent > 0),
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[32, 16],
        encoder_hidden_dims=[32, 16],
        decoder_hidden_dims=[16, 32],
        num_latent=num_latent,
        distribution_cfg=_make_dist_cfg(),
    )

    policy_param_ids = {id(p) for p in model.policy_parameters()}
    encoder_param_ids = {id(p) for p in model.encoder_parameters()}
    assert policy_param_ids.isdisjoint(encoder_param_ids)

    # 1. Policy forward/backward must NOT produce gradients on encoder/decoder.
    model.zero_grad()
    actions = model(obs, stochastic_output=True)
    loss_actor = actions.pow(2).mean() + model.get_output_log_prob(actions.detach()).mean()
    loss_actor.backward()
    assert any(p.grad is not None for p in model.policy_parameters())
    assert all(p.grad is None for p in model.encoder_parameters())

    # 2. Encoder loss backward must NOT produce gradients on policy MLP/distribution.
    model.zero_grad()
    next_prop = torch.randn(NUM_ENVS, CRITIC_PROP_DIM) if num_latent > 0 else None
    enc_loss, loss_dict = model.compute_encoder_loss(obs, next_prop)
    assert "vel_loss" in loss_dict and "encoder_total_loss" in loss_dict
    enc_loss.backward()
    assert any(p.grad is not None for p in model.encoder_parameters())
    assert all(p.grad is None for p in model.policy_parameters())


@pytest.mark.parametrize("model_cls", [AEModel, VAEModel])
@pytest.mark.parametrize("num_latent", [0, 8])
@pytest.mark.parametrize("obs_normalization", [False, True])
@pytest.mark.parametrize("dist_class", ["GaussianDistribution", "HeteroscedasticGaussianDistribution"])
@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
def test_jit_and_onnx_single_input_export(
    model_cls: type[AEModel] | type[VAEModel],
    num_latent: int,
    obs_normalization: bool,
    dist_class: str,
) -> None:
    obs = _make_estimator_obs()
    model = model_cls(
        obs=obs,
        obs_groups=_make_obs_groups(include_prop=num_latent > 0),
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[32, 16],
        encoder_hidden_dims=[32, 16],
        decoder_hidden_dims=[16, 32],
        num_latent=num_latent,
        obs_normalization=obs_normalization,
        distribution_cfg=_make_dist_cfg(dist_class),
    )
    if obs_normalization:
        model.train()
        for _ in range(5):
            model.update_normalization(_make_estimator_obs(shift=3.0))
    model.eval()

    expected_actions = model(obs).detach()
    norm_hist = model.obs_normalizer(model.get_actor_obs(obs))
    if model_cls is VAEModel:
        expected_vel, expected_latent, _, _ = model.encoder_forward(norm_hist)
        latent_key = "latent_mean"
    else:
        expected_vel, expected_latent = model.encoder_forward(norm_hist)
        latent_key = "latent"

    single_input = obs["policy"].reshape(NUM_ENVS, -1)
    jit_model = torch.jit.script(model.as_jit())
    jit_outputs = jit_model(single_input)
    torch.testing.assert_close(jit_outputs[0], expected_actions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jit_outputs[1], expected_vel, atol=1e-5, rtol=1e-5)
    if num_latent > 0:
        assert len(jit_outputs) == 3
        torch.testing.assert_close(jit_outputs[2], expected_latent, atol=1e-5, rtol=1e-5)
    else:
        assert len(jit_outputs) == 2

    onnx_model = model.as_onnx()
    onnx_model.eval()
    expected_output_names = ["actions", "est_vel", latent_key] if num_latent > 0 else ["actions", "est_vel"]
    assert onnx_model.input_names == ["obs"]
    assert onnx_model.output_names == expected_output_names

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            f.name,
            export_params=True,
            opset_version=18,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            external_data=False,
        )
        loaded = onnx.load(f.name)
        onnx.checker.check_model(loaded)
        assert [i.name for i in loaded.graph.input] == ["obs"]
        assert [o.name for o in loaded.graph.output] == expected_output_names
