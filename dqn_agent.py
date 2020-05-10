import numpy as np
import random
from collections import namedtuple, deque
import heapq

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:

                self.learn(GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        indices = self.memory.sample()
        experiences = self.memory.indices_to_experiences(indices)
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            priorities = np.concatenate(abs(Q_targets - Q_expected).numpy())
            self.memory.update_priorities(indices, priorities)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        self.min_priority = -1
        self.total_priority = 0

    def compute_partitions(self, alpha=.7):
        partitions = []
        N = len(self.memory)

        # How much rank we're aiming for.
        total_rank = 0
        for i in range(N):
            total_rank += np.power(1/(i + 1), alpha)

        partition_prob = 1/self.batch_size
        target_prob = partition_prob

        cumulative_prob = 0
        for i in range(N):
            rank = i + 1
            cumulative_prob += np.power(1/rank, alpha) / total_rank
            if (cumulative_prob >= target_prob):
                partitions.append(i)
                if len(partitions) == self.batch_size - 1:
                    break
                target_prob = cumulative_prob + partition_prob

        self.partitions = partitions

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        # Heap requires a priority as the first tuple element. Use the memory index as a tie breaker.
        e = (self.min_priority, len(self.memory), (state, action, reward, next_state, done))
        self.memory.insert(0, e)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            # Heap is a min-heap, so invert the priority
            priority = -priority
            if priority < self.min_priority:
                self.min_priority = priority
            self.memory[index] = (priority, self.memory[index][1], self.memory[index][2])

        # This probably shouldn't be run every time the priorities are updated
        heapq.heapify(self.memory)

    def sample(self, alpha=0):
        """Randomly sample a batch of indices from memory."""
        self.compute_partitions()
        N = len(self.memory)
        indices = []
        start = 0

        for end in self.partitions:
            if start == end:
                indices.append(end)
            else:
                indices.append(random.choice(range(start, end)))
            start = end + 1

        if start == N:
            indices.append(N-1)
        else:
            indices.append(random.choice(range(start, N)))

        if len(indices) < self.batch_size:
            indices.extend(random.sample(range(N), self.batch_size - len(indices)))

        return indices

    def indices_to_experiences(self, indices):
        experiences = [self.memory[i][2] for i in indices]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
