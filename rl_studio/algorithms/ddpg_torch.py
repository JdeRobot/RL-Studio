import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils import tensorboard
import pickle

import random
from collections import deque
from collections.abc import Sequence

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size=None, output_size=None, actions_space=None, hidden_size=None, learning_rate=1e-4):
        super(Actor, self).__init__()
        if hidden_size is not None:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, output_size)
            self.actor_optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if actions_space is not None:
            self.noise = OUNoise(actions_space)

    # def train(self, w, advantage, global_steps):
    #     critic_loss = advantage.pow(2).mean()
    #     w.add_scalar("loss/critic_loss", critic_loss, global_step=global_steps)
    #     self.adam_critic.zero_grad()
    #     critic_loss.backward()
    #     # clip_grad_norm_(adam_critic, max_grad_norm)
    #     w.add_histogram("gradients/critic",
    #                         torch.cat([p.data.view(-1) for p in self.parameters()]), global_step=global_steps)
    #     self.adam_critic.step()
    #     return critic_loss

    def reset_noise(self):
        self.noise.reset()

    def get_action(self, state, step=None, explore=True):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.forward(state)
        action = action.detach().numpy()[0, 0]
        if explore:
            action = self.noise.get_action(action, step)

        return action if isinstance(action, Sequence) else [action]

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

    def load_model(self, actor_file_path):
        model = open(actor_file_path, "rb")

        loaded_brain = pickle.load(model)
        self.linear1 = loaded_brain.linear1
        self.linear2 = loaded_brain.linear2
        self.linear3 = loaded_brain.linear3

        print(f"\n\nMODEL LOADED.")

    def inference(self, state):
        return self.get_action(state, explore=False)
