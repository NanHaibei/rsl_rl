import copy
import os
import torch
from rsl_rl.modules import ActorCriticEstNet, ActorCriticDWAQ
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

def export_policy_as_onnx(
    policy: object, path: str, policy_type: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if policy_type == "ActorCritic":
        policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
        policy_exporter.export(path, filename)
    elif policy_type == "ActorCriticEstNet":
        policy_exporter = EstNetOnnxPolicyExporter(policy, normalizer=normalizer)
        policy_exporter.export(path=path, filename="Estnet_policy.onnx")
    elif policy_type == "ActorCriticDWAQ":
        policy_exporter = DWAQOnnxPolicyExporter(policy, normalizer=normalizer)
        policy_exporter.export(path=path, filename="Estnet_policy.onnx")

class EstNetOnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy: ActorCriticEstNet, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)

        # copy encoder
        self.encoder = copy.deepcopy(policy.encoder)
        # 一帧观测的长度
        self.obs_one_frame_len = policy.obs_one_frame_len

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        obs = self.normalizer(x)
        est_vel = self.encoder(x)
        new_obs = obs[:, 0:self.obs_one_frame_len]
        obs_actor = torch.cat((est_vel.detach(), new_obs), dim=-1)
        actions_mean = self.actor(obs_actor)
        return actions_mean, est_vel

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        obs = torch.zeros(1, self.encoder[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions","est_vel"],
            dynamic_axes={},
        )

class DWAQOnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy: ActorCriticDWAQ, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)

        # copy encoder
        self.encoder = copy.deepcopy(policy.encoder_backbone)
        self.encoder_vel_head = copy.deepcopy(policy.encoder_vel_mean)
        self.encoder_latent_head = copy.deepcopy(policy.encoder_latent_mean)
        # 一帧观测的长度
        self.obs_one_frame_len = policy.obs_one_frame_len
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
    
    def forward(self, obs):
        obs_noarmalized = self.normalizer(obs)
        x = self.encoder(obs_noarmalized)
        vel = self.encoder_vel_head(x)
        latent = self.encoder_latent_head(x)
        code = torch.cat((vel, latent), dim=-1)
        now_obs = obs_noarmalized[:, 0:self.obs_one_frame_len]  # 获取当前观测值
        observations = torch.cat((code.detach(), now_obs), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean, vel
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        obs = torch.zeros(1, self.encoder[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions","est_vel"],
            dynamic_axes={},
        )