# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticEstNet, ActorCriticDWAQ
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, ReplayBuffer
from rsl_rl.utils import string_to_callable, Normalizer
from typing import Any, NoReturn
from collections import deque
from rsl_rl.utils import AMPLoader

class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstNet | ActorCriticDWAQ
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstNet | ActorCriticDWAQ,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Training configuration (从 runner 传递)
        train_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            # TODO为什么不支持
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstNet | ActorCriticDWAQ = policy
        self.policy.to(self.device)
        # 如果使用了EstNet与DWAQ
        self.estnet = False
        self.dwaq = False
        self.estnet = True if type(self.policy) == ActorCriticEstNet else False
        self.dwaq = True if type(self.policy) == ActorCriticDWAQ else False
        # self.delta_sine = True if type(self.policy) == ActorCritic_DeltaSine else False

        # Create optimizer
        if self.estnet:
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters()},
                {'params': self.policy.critic.parameters()},
                {'params': [self.policy.std] if self.policy.noise_std_type == "scalar" else [self.policy.log_std]},
            ], lr=learning_rate)
            self.encoder_optimizer = torch.optim.Adam([
                {'params': self.policy.encoder.parameters()},
            ], lr=learning_rate)
        elif self.dwaq:
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters()},
                {'params': self.policy.critic.parameters()},
                {'params': [self.policy.std] if self.policy.noise_std_type == "scalar" else [self.policy.log_std]},
            ], lr=learning_rate)
            self.encoder_optimizer = torch.optim.Adam([
                {'params': self.policy.encoder_backbone.parameters()},
                {'params': self.policy.encoder_latent_mean.parameters()},
                {'params': self.policy.encoder_latent_logvar.parameters()},
                {'params': self.policy.encoder_vel_mean.parameters()},
                {'params': self.policy.encoder_vel_logvar.parameters()},
                {'params': self.policy.decoder.parameters()},
            ], lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # AMP Discriminator
        if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None:
            amp_cfg = train_cfg["amp_cfg"]
            self.amp_storage = ReplayBuffer(self.policy.amp_discriminator.input_dim // 2, 100000, device)

            # TODO:使用不同的学习率
            self.discriminator_optimizer = torch.optim.Adam( self.policy.amp_discriminator.parameters(),lr=amp_cfg["discr_learning_rate"],)
            
            # AMP Discriminator parameters
            self.disc_update_decimation = amp_cfg["discr_update_decimation"]
            self.disc_update_counter = 0  # 用于跟踪mini-batch计数

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions, extra_info = self.policy.act(obs)
        self.transition.actions = self.transition.actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions, extra_info

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # 如果使用了AMP则重新计算奖励
        if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None:
            # 保存原始 task reward
            self.task_rewards = rewards.clone()
            
            # 使用 self.transition.observations 作为当前观测，obs 参数作为下一个观测
            amp_obs = self.transition.observations["amp"]
            next_amp_obs = self.transition.next_observations["amp"]
            
            # 计算 style reward（判别器输出）
            self.final_rewards, self.style_rewards, _, _, _ = self.policy.amp_discriminator.predict_amp_reward(
                amp_obs, next_amp_obs, self.task_rewards
            )
            
            # 使用 final reward 作为最终奖励
            rewards = self.final_rewards
            # 保存amp policy观测值
            self.amp_storage.insert(amp_obs, next_amp_obs)
        else:
            self.task_rewards = None
            self.style_rewards = None
            self.final_rewards = None
            
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # -- vel est loss
        mean_vel_loss = 0 if self.estnet or self.dwaq else None
        # -- DWAQ loss
        mean_obs_loss = 0 if self.dwaq else None
        mean_dkl_loss = 0 if self.dwaq else None
        mean_dwaq_loss = 0 if self.dwaq else None
        # -- AMP loss
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        disc_actual_updates = 0  # 记录判别器实际更新的次数

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 只在使用AMP时创建额外的生成器
        if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            amp_expert_generator = self.policy.amp_expert_data.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            # 使用zip同时遍历三个生成器
            combined_generator = zip(generator, amp_policy_generator, amp_expert_generator)
        else:
            # 不使用AMP时,只包装generator以保持统一的接口
            combined_generator = ((sample, None, None) for sample in generator)

        # 从环境采样的数据 + AMP 策略样本 + AMP 专家样本 同时取出一批
        for sample, sample_amp_policy, sample_amp_expert in combined_generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
                next_observations_batch
            ) = sample

            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    if self.estnet or self.dwaq:
                        for param_group in self.encoder_optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

            # Estimate Net step
            if self.estnet:
                policy_obs = self.policy.get_actor_obs(obs_batch)
                policy_obs = self.policy.actor_obs_normalizer(policy_obs)
                vel_est = self.policy.encoder_forward(policy_obs) 
                vel_target = self.policy.get_critic_obs(obs_batch)[:,0:3]
                vel_target.requires_grad = False
                vel_MSE = nn.MSELoss()(vel_est, vel_target) * 1000.0 # 小数使用L2loss后太小了，没有梯度

                self.encoder_optimizer.zero_grad()
                vel_MSE.backward(retain_graph=True)
                encoder_params = [p for group in self.encoder_optimizer.param_groups for p in group['params']]
                grad_norm = nn.utils.clip_grad_norm_(encoder_params, self.max_grad_norm)
                self.encoder_optimizer.step()

            # DWAQ step TODO:使用vel_mean计算MSE、仅使用zt计算KL散度
            if self.dwaq:
                policy_obs = self.policy.get_actor_obs(obs_batch)
                policy_obs = self.policy.actor_obs_normalizer(policy_obs)
                code,vel_sample,decode,vel_mean,vel_logvar,latent_mean,latent_logvar = self.policy.encoder_forward(policy_obs) 
                vel_target = self.policy.get_critic_obs(obs_batch)[:,0:3]
                next_observations = self.policy.get_actor_obs(next_observations_batch)
                next_observations = self.policy.actor_obs_normalizer(next_observations)
                obs_target = next_observations[:, 0:self.policy.obs_one_frame_len]  # 取最新一帧obs
                
                vel_target.requires_grad = False
                obs_target.requires_grad = False
                # DreamWaQ损失=速度重建损失 + obs重建损失 + KL散度损失
                vel_MSE = nn.MSELoss()(vel_sample, vel_target) * 10.0 # TODO:应该使用均值而不是采样值
                obs_MSE = nn.MSELoss()(decode, obs_target) * 10.0 # TODO:应该使用均值而不是采样值
                # KL散度损失：按批次平均
                dkl_loss = -0.5 * torch.mean(torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1))
                autoenc_loss = vel_MSE + obs_MSE + self.policy.beta * dkl_loss 
                self.encoder_optimizer.zero_grad()
                autoenc_loss.backward(retain_graph=True)
                encoder_params = [p for group in self.encoder_optimizer.param_groups for p in group['params']]
                grad_norm = nn.utils.clip_grad_norm_(encoder_params, self.max_grad_norm)  # 使用更小的梯度裁剪阈值
                self.encoder_optimizer.step()

            # Discriminator loss
            # 根据 disc_update_decimation 控制判别器更新频率，提前判断避免不必要的计算
            if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None and (self.disc_update_counter % self.disc_update_decimation == 0):
                # 获取amp的policy和expert数据
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                # 如果使用了AMP归一化器
                if self.policy.amp_discriminator.normalizer is not None:
                    # 更新归一化器的参数
                    self.policy.amp_discriminator.normalizer.update(policy_state.cpu().numpy())
                    self.policy.amp_discriminator.normalizer.update(expert_state.cpu().numpy())
                    # 对数据进行归一化
                    with torch.no_grad():
                        policy_state = self.policy.amp_discriminator.normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.policy.amp_discriminator.normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.policy.amp_discriminator.normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.policy.amp_discriminator.normalizer.normalize_torch(expert_next_state, self.device)

                # 计算AMP损失
                policy_d = self.policy.amp_discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.policy.amp_discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.policy.amp_discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10.0)
                self.amploss_coef=1.0
                discriminator_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

                # 更新判别器
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.amp_discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()
                disc_actual_updates += 1  # 记录实际更新次数

                # Store the losses
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

            # 更新计数器（无论是否更新判别器都要更新）
            if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None:
                self.disc_update_counter += 1


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch, _ = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # vel est loss
            if mean_vel_loss is not None:
                mean_vel_loss += vel_MSE.item()
            # DWAQ loss
            if mean_obs_loss is not None:
                mean_obs_loss += obs_MSE.item()
            if mean_dkl_loss is not None:
                mean_dkl_loss += dkl_loss.item()
            if mean_dwaq_loss is not None:
                mean_dwaq_loss += autoenc_loss.item()
            # AMP loss - 已在判别器更新时累积，这里不需要重复累积

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For vel est
        if mean_vel_loss is not None:
            mean_vel_loss /= num_updates
        # -- For DWAQ
        if mean_vel_loss is not None:
            mean_vel_loss /= num_updates
        if mean_obs_loss is not None:
            mean_obs_loss /= num_updates
        if mean_dkl_loss is not None:
            mean_dkl_loss /= num_updates
        if mean_dwaq_loss is not None:
            mean_dwaq_loss /= num_updates
        # -- For AMP (使用实际更新次数进行平均，如果没有更新则除以num_updates)
        if mean_amp_loss:
            # 如果判别器有实际更新，使用实际更新次数；否则使用总批次数
            amp_divisor = disc_actual_updates if disc_actual_updates > 0 else num_updates
            mean_amp_loss /= amp_divisor
            mean_grad_pen_loss /= amp_divisor
            mean_policy_pred /= amp_divisor
            mean_expert_pred /= amp_divisor
        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.estnet or self.dwaq:
            loss_dict["vel_loss"] = mean_vel_loss
        if self.dwaq:
            loss_dict["obs_loss"] = mean_obs_loss
            loss_dict["dkl_loss"] = mean_dkl_loss
            loss_dict["dwaq_loss"] = mean_dwaq_loss
        if hasattr(self.policy, 'amp_discriminator') and self.policy.amp_discriminator is not None:
            loss_dict["amp"] = mean_amp_loss
            loss_dict["amp_grad_pen"] = mean_grad_pen_loss
            loss_dict["amp_policy_pred"] = mean_policy_pred
            loss_dict["amp_expert_pred"] = mean_expert_pred

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
