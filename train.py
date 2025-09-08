#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import argparse
from datetime import datetime

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import json
import dmc2gym


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent='sac')  # Hardcoded agent as 'sac'

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        # Manually instantiate actor, critic, and agent (replacing hydra.utils.instantiate)
        from agent.actor import DiagGaussianActor
        actor_log_std_bounds = [int(x) for x in cfg.actor_log_std_bounds.split(',')]
        actor = DiagGaussianActor(obs_dim, action_dim, cfg.actor_hidden_dim, cfg.actor_hidden_depth, actor_log_std_bounds)

        from agent.critic import DoubleQCritic
        critic = DoubleQCritic(obs_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_hidden_depth)

        from agent.sac import SACAgent
        self.agent = SACAgent(obs_dim, action_dim, action_range, self.device, critic, actor,
                              discount=cfg.discount,
                              init_temperature=cfg.init_temperature,
                              alpha_lr=cfg.alpha_lr,
                              alpha_betas=[float(x) for x in cfg.alpha_betas.split(',')],
                              actor_lr=cfg.actor_lr,
                              actor_betas=[float(x) for x in cfg.actor_betas.split(',')],
                              actor_update_frequency=cfg.actor_update_frequency,
                              critic_lr=cfg.critic_lr,
                              critic_betas=[float(x) for x in cfg.critic_betas.split(',')],
                              critic_tau=cfg.critic_tau,
                              critic_target_update_frequency=cfg.critic_target_update_frequency,
                              batch_size=cfg.batch_size,
                              learnable_temperature=cfg.learnable_temperature)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


def main():
    parser = argparse.ArgumentParser(description='SAC Training Script')

    # Parameters from train.yaml
    parser.add_argument('--env', type=str, default='cheetah_run', help='Environment name')
    parser.add_argument('--experiment', type=str, default='test_exp', help='Experiment name')
    parser.add_argument('--num_train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--replay_buffer_capacity', type=int, default=None, help='Replay buffer capacity (defaults to num_train_steps)')
    parser.add_argument('--num_seed_steps', type=int, default=5000, help='Number of seed steps')
    parser.add_argument('--eval_frequency', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--log_frequency', type=int, default=10000, help='Logging frequency')
    parser.add_argument('--log_save_tb', action='store_true', default=True, help='Save TensorBoard logs')
    parser.add_argument('--save_video', action='store_true', default=True, help='Save evaluation videos')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # SAC-specific parameters from sac.yaml
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--init_temperature', type=float, default=0.1, help='Initial temperature')
    parser.add_argument('--alpha_lr', type=float, default=1e-4, help='Alpha learning rate')
    parser.add_argument('--alpha_betas', type=str, default='0.9,0.999', help='Alpha betas (comma-separated)')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--actor_betas', type=str, default='0.9,0.999', help='Actor betas (comma-separated)')
    parser.add_argument('--actor_update_frequency', type=int, default=1, help='Actor update frequency')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='Critic learning rate')
    parser.add_argument('--critic_betas', type=str, default='0.9,0.999', help='Critic betas (comma-separated)')
    parser.add_argument('--critic_tau', type=float, default=0.005, help='Critic tau')
    parser.add_argument('--critic_target_update_frequency', type=int, default=2, help='Critic target update frequency')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learnable_temperature', action='store_true', default=True, help='Learnable temperature')

    # Actor and Critic architecture parameters
    parser.add_argument('--actor_hidden_dim', type=int, default=1024, help='Actor hidden dimension')
    parser.add_argument('--actor_hidden_depth', type=int, default=2, help='Actor hidden depth')
    parser.add_argument('--actor_log_std_bounds', type=str, default='-5,2', help='Actor log std bounds (comma-separated)')
    parser.add_argument('--critic_hidden_dim', type=int, default=1024, help='Critic hidden dimension')
    parser.add_argument('--critic_hidden_depth', type=int, default=2, help='Critic hidden depth')

    args = parser.parse_args()

    # Simulate YAML interpolation: replay_buffer_capacity defaults to num_train_steps
    if args.replay_buffer_capacity is None:
        args.replay_buffer_capacity = args.num_train_steps

    # Create output directory similar to Hydra
    now = datetime.now()
    output_dir = f"./exp/{now.strftime('%Y.%m.%d')}/{now.strftime('%H%M')}_sac_{args.experiment}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    os.chdir(output_dir)

    workspace = Workspace(args)
    workspace.run()


if __name__ == '__main__':
    main()