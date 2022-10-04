import copy
import pickle
import random
from collections import deque

import torch
from torch import nn


class DQN_Agent:
    def __init__(
            self, layer_sizes, lr=1e-3, sync_freq=5, exp_replay_size=256, seed=1423, gamma=0, block_batch=False
    ):

        self.exp_replay_size = exp_replay_size
        self.block_batch = block_batch
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.cuda()
        self.target_net.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(gamma).float().cuda()
        if block_batch:
            self.blocked_batch = deque(maxlen=int(exp_replay_size / 4))
            self.experience_replay = deque(maxlen=int(3 * exp_replay_size / 4))
        else:
            self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().cuda())
        Q, A = torch.max(Qp, axis=0)
        A = (A if torch.rand(1,).item() >= epsilon
            else torch.randint(0, action_space_len, (1,)))
        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        if self.block_batch and len(self.blocked_batch) < self.exp_replay_size / 4:
            self.blocked_batch.append(experience)
        else:
            self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if self.block_batch and random.uniform(0, 1) >= 0.25:
            batch = self.blocked_batch
        else:
            batch = self.experience_replay

        if len(batch) < sample_size:
            sample_size = len(batch)
        sample = random.sample(batch, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.cuda())
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.cuda())
        target_return = rn.cuda() + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()

    def inference(self, state):
        return self.get_action(state, None, epsilon=0)

    def load_model(self, weights_file_path):

        qnet_weights = open(weights_file_path, "rb")

        self.q_net = pickle.load(qnet_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        print(f"\n\nMODEL LOADED.")
        print(f"    - Loading:    {weights_file_path}")
        print(f"    - Model size: {len(self.q_net)}")
