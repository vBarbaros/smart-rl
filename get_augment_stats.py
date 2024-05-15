"""
Copyright 2024 Victor Barbaros. All Rights Reserved.

Code adapted from:
https://github.com/facebookresearch/drqv2
"""

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
# os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from dm_env import specs
import ot
from piq import ssim

import utils.dmc as dmc
import utils.utils as utils
from utils.metrics import StatsMetrics
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def make_agent(obs_spec, action_spec, cfg):
    cfg.agent.obs_shape = obs_spec.shape
    if cfg.discrete_actions:
        cfg.agent.num_actions = action_spec.num_values
    else:
        cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.stats = StatsMetrics()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # some assertions
        utils.assert_agent(self.cfg['agent']['agent_name'], self.cfg['pixel_obs'])

        # create logger
        self.logger = Logger(self.work_dir)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed,
                                  self.cfg.pixel_obs, self.cfg.discrete_actions)

        self.obsstats = self.work_dir / 'obsstats'
        self.obsstats.mkdir(exist_ok=True)

        # save cfg
        utils.save_cfg(self.cfg, self.work_dir)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_augment_episodes)

        while eval_until_episode(episode):
            time_step = self.train_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, obs_tensor, obs_images, obs_aug_tensor, obs_aug_images = self.agent.act_with_metrics(
                        time_step.observation, self.global_step, eval_mode=True, save_image=True)

                sm = self.stats.get_metrics(obs_tensor, obs_aug_tensor)
                time_step = self.train_env.step(action)
                step += 1
                self._global_step += 1

                with self.logger.log_and_dump_ctx(self.global_frame, ty='augment') as log:
                    log('ssim_dist', sm['ssim_dist'])
                    log('hamming', sm['hamming'])
                    log('bhattacharyya', sm['bhattacharyya'])
                    log('kl_div', sm['kl_div'])
                    log('chebyshev', sm['chebyshev'])
                    log('manhattan', sm['manhattan'])
                    log('euclidian', sm['euclidian'])
                    log('cosine_dist', sm['cosine_dist'])
                    log('mu_original', sm['mu_original'])
                    log('mu_augment', sm['mu_augment'])
                    log('sigma_original', sm['sigma_original'])
                    log('sigma_augment', sm['sigma_augment'])
                    log('step', self.global_step)

            episode += 1



    def log_augment_obs_stats(self, task_id=1):
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # try to evaluate
        if eval_every_step(self.global_step):
            self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
            self.eval()


def save_snapshot(self):
    snapshot = self.work_dir / 'snapshot.pt'
    keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
    payload = {k: self.__dict__[k] for k in keys_to_save}
    with snapshot.open('wb') as f:
        torch.save(payload, f)


def load_snapshot(self):
    snapshot = self.work_dir / 'snapshot.pt'
    with snapshot.open('rb') as f:
        payload = torch.load(f)
    for k, v in payload.items():
        self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from get_augment_stats import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.log_augment_obs_stats()


if __name__ == '__main__':
    main()
    # ==> $ python train.py task=pendulum_swingup agent=drqv2_pad_2
