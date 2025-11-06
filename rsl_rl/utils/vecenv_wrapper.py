from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

class RslRlVecEnvWrapperDictAction(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        super().__init__(env, clip_actions)

    def step(self, actions: torch.Tensor, extra_info) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # 创建字典action
        action_dict = {}
        action_dict["action"] = actions
        action_dict["extra_info"] = extra_info
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(action_dict)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
