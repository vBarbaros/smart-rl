"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.

Official Implementation of DRQ-v2
https://arxiv.org/abs/2107.09645

Code is based on:
https://github.com/facebookresearch/drqv2
"""

import hydra
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from models.cnn import PixelEncoder
from models.core import DrQActor, Critic
import utils.utils as utils

from utils.augmenter import AugmentationFactory


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim,
                 linear_approx, critic_target_tau, num_expl_steps, update_every_steps,
                 stddev_schedule, stddev_clip, augment_type, augment_param):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(device)
        self.actor = DrQActor(feature_dim, action_shape[0], hidden_dim).to(device)

        self.critic = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.pixel_encoder_opt = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        if augment_type == 'rotate' and augment_param is not None:
            self.aug = AugmentationFactory(augmentation_type='rotate', rotate_angle=augment_param)
        elif augment_type == 'shift' and augment_param is not None:
            self.aug = AugmentationFactory(augmentation_type='shift', pad=augment_param)
        elif augment_type == 'contrast' and augment_param is not None:
            self.aug = AugmentationFactory(augmentation_type='contrast', contrast_factor=augment_param)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.pixel_encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.pixel_encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def act_with_metrics(self, obs, step, eval_mode, save_image=False):
        # augment
        original_images = [to_pil_image(obs[i]) for i in range(obs.shape[0])]

        if save_image:
            for idx, img in enumerate(original_images):
                img.save(f'image_{idx}.png')

        # encode
        obs = torch.as_tensor(obs, device=self.device)

        obs_pre_aug = obs.clone().unsqueeze(1)
        obs_orig_aug = obs.clone()
        obs_aug = self.aug(obs_pre_aug.float())

        obs_augment_tensor = obs_aug.clone().squeeze(0)
        augmented_images = [to_pil_image(obs_augment_tensor[i]) for i in range(obs_augment_tensor.shape[0])]
        if save_image:
            for idx, img in enumerate(augmented_images):
                img.save(f'aug_image_{idx}.png')

        obs = self.pixel_encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0], obs_orig_aug, original_images, obs_augment_tensor, augmented_images

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        obs = self.pixel_encoder(obs)
        q, _ = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.pixel_encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.pixel_encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.pixel_encoder(obs)
        with torch.no_grad():
            next_obs = self.pixel_encoder(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        torch.save(self.pixel_encoder.state_dict(), f'{model_save_dir}/pixel_encoder.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
        self.pixel_encoder.load_state_dict(
            torch.load(f'{model_load_dir}/pixel_encoder.pt', map_location=self.device)
        )
