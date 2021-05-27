from collections import namedtuple, deque
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.transform import resize
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('workon', device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'final'))


class Memory(object):
    def __init__(self, capacity: int):
        # when new items are added, a corresponding number
        # of items are discarded from the opposite end.
        self.memory = deque([], maxlen=capacity)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        total = len(self.memory)
        transitions = []
        if(total < batch_size):
            divide = math.floor(batch_size / total)
            residue = batch_size % total
            residue_transitions = random.sample(self.memory, residue)
            for i in range(divide):
                transitions += random.sample(self.memory, total)
            transitions += residue_transitions
        else:
            transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class DQNAgent(object):
    def __init__(self, net, net_args, n_actions):
        self.n_actions = n_actions
        self.batch_size = 100
        self.step_count = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 500
        self.gammay = 0.99
        self.target_update_steps = 300
        self.eval_net = net(*net_args).to(device)
        self.target_net = net(*net_args).to(device)
        self.memory = Memory(1000)
        self.optimizer = optim.Adam(self.eval_net.parameters(), 0.01)
        self.criterion = nn.SmoothL1Loss()

    @property
    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step_count / self.eps_decay)

    def choose_action(self, state: np.ndarray):
        if random.random() < self.eps:
            x = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(self.eval_net(x), 1).item()
        else:
            action = np.random.randint(0, self.n_actions)

        self.step_count += 1
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        final_batch = torch.BoolTensor(batch.final).to(device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(device)  # shape (batch, 3, w, h)
        state_batch = torch.FloatTensor(batch.state).to(device)  # shape (batch, 3, w, h)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)  # shape (batch, 1)
        reward_batch = torch.FloatTensor(batch.reward).to(device)  # shape (batch, 1)
        # print('final_mask', final_batch.shape)
        # print('state_batch', state_batch.shape)
        # print('action_batch', action_batch.shape)
        # print('reward_batch', reward_batch.shape)

        q = self.eval_net(state_batch).gather(1, action_batch)  # shape (batch, 1)
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        next_q[final_batch] = 0
        target_q = reward_batch + next_q * self.gammay
        target_q = target_q.unsqueeze(1)  # shape (batch, 1)

        loss = self.criterion(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.step_count % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return loss.item()
