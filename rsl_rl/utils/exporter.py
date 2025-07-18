# 编写各种网络结构导出onnx的函数
import copy
import os
import torch
from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ


class DWAQOnnxPolicyExporter(torch.nn.Module):
    def __init__(
        self, 
        policy: ActorCritic_DWAQ, 
        path: str, 
        ac_normalizer: object | None = None,
        vae_normalizer: object | None = None,
        file_name: str = "dwaq.onnx",
        verbose=False
    ):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.encoder_vel_head = copy.deepcopy(policy.encode_mean_vel)
        self.encoder_latent_head = copy.deepcopy(policy.encode_mean_latent)

        self.vae_normalizer = copy.deepcopy(vae_normalizer)
        self.ac_normalizer = copy.deepcopy(ac_normalizer)

        self.path = path
        self.file_name = file_name

    def forward(self, obs, obs_his):
        x = self.encoder(self.vae_normalizer(obs_his))
        mean_vel = self.encoder_vel_head(x)
        mean_latent = self.encoder_latent_head(x)
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        observations = torch.cat((code.detach(), self.ac_normalizer(obs)), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean, mean_vel
    
    def export(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        self.to("cpu")
        
        code_lenth = self.encoder_vel_head.out_features + self.encoder_latent_head.out_features
        obs_his = torch.zeros(1, self.encoder[0].in_features)
        obs = torch.zeros(1, self.actor[0].in_features - code_lenth)
        torch.onnx.export(
            self,
            (obs, obs_his),
            os.path.join(self.path, self.file_name),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "obs_his"],
            output_names=["actions_mean", "mean_vel"],
            dynamic_axes={},
        )
