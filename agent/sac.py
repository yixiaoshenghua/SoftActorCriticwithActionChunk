import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy  # Added for deepcopy

from agent import Agent
import utils


class SACAgent(Agent):
    """SAC algorithm with Q-chunking (temporally extended action space)."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic, actor,
                 discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, chunk_size, confidence_threshold=0.5):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.chunk_size = chunk_size
        self.action_dim = action_dim  # Original action_dim
        self.extended_action_dim = action_dim * chunk_size  # Extended for chunk
        self.confidence_threshold = confidence_threshold

        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(critic).to(self.device)  # Create a deep copy
        self.actor = actor.to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A| * h (extended action space)
        self.target_entropy = -self.extended_action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist, confidences = self.actor(obs)  # dist is over the entire chunk, confidences (1, h)
        if sample:
            chunk_action = dist.rsample()  # Sample entire chunk
        else:
            chunk_action = dist.mean  # Mean for eval
        chunk_action = chunk_action.clamp(self.action_range[0], self.action_range[1])
        chunk_action = chunk_action.reshape((self.chunk_size, self.action_dim))
        return utils.to_np(chunk_action), utils.to_np(confidences)

    def _get_length(self, confidences):
        batch_size = confidences.shape[0]
        lengths = torch.full((batch_size,), self.chunk_size, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            for i in range(self.chunk_size):
                if confidences[b, i] < self.confidence_threshold:
                    lengths[b] = max(1, i + 1)
                    break
        return lengths

    def update_critic(self, state, action_chunk, reward_chunk, next_state, not_done, logger, step, lengths, masks):
        # action_chunk: [batch, h * action_dim], padded
        # reward_chunk: [batch, h, 1], padded
        # lengths: [batch]
        # masks: [batch, h]

        dist, confidences = self.actor(next_state)
        next_action_chunk_full = dist.rsample()  # [batch, h * action_dim]
        log_prob = dist.log_prob(next_action_chunk_full).sum(-1, keepdim=True)  # Over entire chunk

        # Compute next_lengths
        next_lengths = self._get_length(confidences)

        # Pad next_action_chunk_full and create masks
        batch_size = next_state.shape[0]
        next_padded_actions = torch.zeros(batch_size, self.extended_action_dim, device=self.device)
        next_masks = torch.zeros(batch_size, self.chunk_size, device=self.device)
        for b in range(batch_size):
            k = next_lengths[b]
            next_padded_actions[b, :k * self.action_dim] = next_action_chunk_full[b, :k * self.action_dim]
            next_masks[b, :k] = 1.0

        target_Q1, target_Q2 = self.critic_target(next_state, next_padded_actions, next_masks)

        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        # Compute discounted multi-step reward sum using actual lengths
        batch_size = state.shape[0]
        h_step_rewards = torch.zeros(batch_size, 1, device=self.device)
        gamma_powers = torch.pow(self.discount, torch.arange(self.chunk_size, dtype=torch.float32, device=self.device)).unsqueeze(0).repeat(batch_size, 1)
        for b in range(batch_size):
            k = lengths[b]
            discounted_rewards = reward_chunk[b, :k] * gamma_powers[b, :k].unsqueeze(1)
            h_step_rewards[b] = discounted_rewards.sum()

        target_Q = h_step_rewards + (not_done * torch.pow(self.discount, lengths.float()).unsqueeze(1) * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action_chunk, masks)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist, confidences = self.actor(obs)
        action_chunk_full = dist.rsample()  # [batch, h * action_dim]
        log_prob = dist.log_prob(action_chunk_full).sum(-1, keepdim=True)  # Over chunk

        # Compute actor_lengths
        actor_lengths = self._get_length(confidences)

        # Pad action_chunk_full and create masks
        batch_size = obs.shape[0]
        padded_actions = torch.zeros(batch_size, self.extended_action_dim, device=self.device)
        actor_masks = torch.zeros(batch_size, self.chunk_size, device=self.device)
        for b in range(batch_size):
            k = actor_lengths[b]
            padded_actions[b, :k * self.action_dim] = action_chunk_full[b, :k * self.action_dim]
            actor_masks[b, :k] = 1.0

        actor_Q1, actor_Q2 = self.critic(obs, padded_actions, actor_masks)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        state, action_chunk, reward_chunk, next_state, not_done, not_done_no_max, lengths, masks = replay_buffer.sample_chunk(
            self.batch_size, self.chunk_size)

        logger.log('train/batch_reward', reward_chunk.mean(), step)

        self.update_critic(state, action_chunk, reward_chunk, next_state, not_done_no_max,
                           logger, step, lengths, masks)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)