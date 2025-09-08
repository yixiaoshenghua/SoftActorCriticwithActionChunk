import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        # For variable chunk sampling
        self.chunk_starts = []  # list of (start_idx, length)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_chunk(self, start_idx, length):
        self.chunk_starts.append((start_idx, length))

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_chunk(self, batch_size, chunk_size):
        """Sample chunks of variable lengths from stored chunk info, pad to fixed chunk_size."""
        if len(self.chunk_starts) == 0:
            # Fallback to original if no chunks
            return self._original_sample_chunk(batch_size, chunk_size)

        # Sample batch_size chunk indices
        chunk_idxs = np.random.randint(0, len(self.chunk_starts), size=batch_size)

        states = []
        action_chunks = []
        reward_chunks = []
        next_states = []
        not_dones = []
        not_dones_no_max = []
        lengths = []

        action_dim = self.actions.shape[1] if len(self.actions.shape) > 1 else 1
        extended_action_dim = chunk_size * action_dim

        for ci in chunk_idxs:
            start, len_k = self.chunk_starts[ci]
            len_k = min(len_k, chunk_size)  # Cap to max

            # Handle circular buffer with %
            action_list = [self.actions[(start + i) % self.capacity] for i in range(len_k)]
            reward_list = [self.rewards[(start + i) % self.capacity] for i in range(len_k)]

            action_chunk = np.concatenate(action_list, axis=0)
            reward_chunk = np.concatenate(reward_list, axis=0)

            # Pad
            padded_action = np.zeros(extended_action_dim, dtype=np.float32)
            padded_action[:len_k * action_dim] = action_chunk.flatten()

            padded_reward = np.zeros(chunk_size, dtype=np.float32)
            padded_reward[:len_k] = reward_chunk.flatten()

            state = self.obses[start % self.capacity]
            next_state = self.next_obses[(start + len_k - 1) % self.capacity]
            not_done = self.not_dones[(start + len_k - 1) % self.capacity]
            not_done_nm = self.not_dones_no_max[(start + len_k - 1) % self.capacity]

            states.append(state)
            action_chunks.append(padded_action)
            reward_chunks.append(padded_reward)
            next_states.append(next_state)
            not_dones.append(not_done)
            not_dones_no_max.append(not_done_nm)
            lengths.append(len_k)

        # Tensorize
        states = torch.as_tensor(np.array(states), device=self.device).float()
        action_chunks = torch.as_tensor(np.array(action_chunks), device=self.device)
        reward_chunks = torch.as_tensor(np.array(reward_chunks), device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(np.array(next_states), device=self.device).float()
        not_dones = torch.as_tensor(np.array(not_dones), device=self.device)
        not_dones_no_max = torch.as_tensor(np.array(not_dones_no_max), device=self.device)
        lengths = torch.as_tensor(lengths, device=self.device)

        # Create masks
        masks = torch.zeros(batch_size, chunk_size, device=self.device)
        for b in range(batch_size):
            masks[b, :lengths[b]] = 1.0

        return states, action_chunks, reward_chunks, next_states, not_dones, not_dones_no_max, lengths, masks

    def sample_fixed_chunk(self, batch_size, chunk_size):
        """Fallback to fixed chunk sampling."""
        max_start = (self.capacity if self.full else self.idx) - chunk_size
        start_idxs = np.random.randint(0, max_start, size=batch_size)

        states = torch.as_tensor(self.obses[start_idxs], device=self.device).float()
        action_chunks = torch.as_tensor(np.concatenate([self.actions[start_idxs + i] for i in range(chunk_size)], axis=1), device=self.device)
        reward_chunks = torch.as_tensor(np.concatenate([self.rewards[start_idxs + i] for i in range(chunk_size)], axis=1), device=self.device)
        next_states = torch.as_tensor(self.next_obses[start_idxs + chunk_size - 1], device=self.device).float()  # st+h
        not_dones = torch.as_tensor(self.not_dones[start_idxs + chunk_size - 1], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[start_idxs + chunk_size - 1], device=self.device)

        lengths = torch.full((batch_size,), chunk_size, device=self.device)
        masks = torch.ones(batch_size, chunk_size, device=self.device)

        return states, action_chunks, reward_chunks, next_states, not_dones, not_dones_no_max, lengths, masks