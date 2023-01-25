import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils import tensorboard
import pickle


def mish(input):
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, input): return mish(input)


# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()

def get_dist(x):
    return torch.distributions.Categorical(probs=x)

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )
        self.adam_actor = torch.optim.Adam(self.parameters(), lr=3e-4)
        torch.manual_seed(1)

    def train(self, w, prev_prob_act, prob_act, advantage, global_steps, epsilon):
        actor_loss = self.policy_loss(prev_prob_act.detach(), prob_act, advantage.detach(), epsilon)
        w.add_scalar("loss/actor_loss", actor_loss, global_step=global_steps)
        self.adam_actor.zero_grad()
        actor_loss.backward()
        # clip_grad_norm_(adam_actor, max_grad_norm)
        # w.add_histogram("gradients/actor",
        #                 torch.cat([p.grad.view(-1) for p in self.parameters()]), global_step=global_steps)
        self.adam_actor.step()
        return actor_loss

    def clip_grad_norm_(module, max_grad_norm):
        nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

    def policy_loss(self, old_log_prob, log_prob, advantage, eps):
        ratio = (log_prob - old_log_prob).exp()
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

        m = torch.min(ratio * advantage, clipped)
        return -m

    def forward(self, X):
        return self.model(X)

    def load_model(self, actor_file_path):
        model = open(actor_file_path, "rb")

        self.model = pickle.load(model)

        print(f"\n\nMODEL LOADED.")

    def inference(self, state):
        probs = self(t(state))
        dist = get_dist(probs)
        action = dist.sample()

        return action


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )
        self.adam_critic = torch.optim.Adam(self.parameters(), lr=1e-3)
        torch.manual_seed(1)

    def train(self, w, advantage, global_steps):
        critic_loss = advantage.pow(2).mean()
        w.add_scalar("loss/critic_loss", critic_loss, global_step=global_steps)
        self.adam_critic.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_(adam_critic, max_grad_norm)
        # w.add_histogram("gradients/critic",
        #                     torch.cat([p.data.view(-1) for p in self.parameters()]), global_step=global_steps)
        self.adam_critic.step()
        return critic_loss

    def forward(self, X):
        return self.model(X)
