# 编写各种网络结构导出onnx的函数
import copy
import os
import torch
from rsl_rl.modules import ActorCritic_DWAQ, ActorCritic_EstNet

class EstNetOnnxPolicyExporter(torch.nn.Module):
    """导出EstNet策略"""
    def __init__(
        self, 
        policy: ActorCritic_EstNet, 
        path: str, 
        obs_normalizer: object | None = None,
        file_name: str = "estnet.onnx",
        verbose=False
    ):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.encoder_vel_head = copy.deepcopy(policy.encode_vel)
        self.encoder_latent_head = copy.deepcopy(policy.encode_latent)
        self.obs_normalizer = copy.deepcopy(obs_normalizer)
        self.path = path
        self.file_name = file_name
        self.latent_len = self.encoder_vel_head.out_features + self.encoder_latent_head.out_features
        self.one_obs_len = self.actor[0].in_features - self.latent_len  # 一帧观测值的长度

    def forward(self, obs):
        obs_noarmalized = self.obs_normalizer(obs)
        x = self.encoder(obs_noarmalized)
        vel = self.encoder_vel_head(x)
        latent = self.encoder_latent_head(x)
        code = torch.cat((vel, latent), dim=-1)
        now_obs = obs_noarmalized[:, -self.one_obs_len:]  # 获取当前观测值
        observations = torch.cat((code.detach(), now_obs), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean, vel
    
    def export(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        self.to("cpu")
        
        obs_his = torch.zeros(1, self.encoder[0].in_features)
        torch.onnx.export(
            self,
            (obs_his),
            os.path.join(self.path, self.file_name),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs_his"],
            output_names=["actions_mean", "mean_vel"],
            dynamic_axes={},
        )


class DWAQOnnxPolicyExporter(torch.nn.Module):
    """导出DWAQ策略,与EstNet导出完全一致"""
    def __init__(
        self, 
        policy: ActorCritic_DWAQ, 
        path: str, 
        obs_normalizer: object | None = None,
        file_name: str = "dwaq.onnx",
        verbose=False
    ):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.encoder_vel_head = copy.deepcopy(policy.encoder_vel_mean)
        self.encoder_latent_head = copy.deepcopy(policy.encoder_latent_mean)
        self.obs_normalizer = copy.deepcopy(obs_normalizer)
        self.path = path
        self.file_name = file_name
        self.latent_len = self.encoder_vel_head.out_features + self.encoder_latent_head.out_features
        self.one_obs_len = self.actor[0].in_features - self.latent_len  # 一帧观测值的长度

    def forward(self, obs):
        obs_noarmalized = self.obs_normalizer(obs)
        x = self.encoder(obs_noarmalized)
        vel = self.encoder_vel_head(x)
        latent = self.encoder_latent_head(x)
        code = torch.cat((vel, latent), dim=-1)
        now_obs = obs_noarmalized[:, -self.one_obs_len:]  # 获取当前观测值
        observations = torch.cat((code.detach(), now_obs), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean, vel
    
    def export(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        self.to("cpu")
        
        obs_his = torch.zeros(1, self.encoder[0].in_features)
        torch.onnx.export(
            self,
            (obs_his),
            os.path.join(self.path, self.file_name),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs_his"],
            output_names=["actions_mean", "mean_vel"],
            dynamic_axes={},
        )
