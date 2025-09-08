import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic for extended action space (chunks)."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, chunk_size):
        super().__init__()

        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.extended_action_dim = action_dim * chunk_size  # Extended dim

        input_dim = obs_dim + self.extended_action_dim
        self.Q1 = utils.mlp(input_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(input_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action_chunk, mask):
        assert obs.size(0) == action_chunk.size(0)
        assert mask.size(1) == self.chunk_size  # mask: [batch, h]

        batch_size = obs.size(0)
        # Expand mask to [batch, h * action_dim]
        expanded_mask = mask.unsqueeze(-1).repeat(1, 1, self.action_dim).view(batch_size, self.extended_action_dim)
        # Mask the action_chunk: non-executed parts become 0
        action_masked = action_chunk * expanded_mask

        obs_action = torch.cat([obs, action_masked], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)