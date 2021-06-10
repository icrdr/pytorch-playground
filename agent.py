from collections import namedtuple, deque
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.transform import resize
import random
import numpy as np
import numpy.random as rd
from net import QNet

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
    def __init__(self, net, net_args, n_actions,
                 batch_size=32,
                 target_update_steps=100,
                 memory_capacity=2000,
                 lr=0.01,
                 gammay=0.9,
                 eps_start=0.9,
                 eps_end=0.1,
                 eps_decay=100):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.step_count = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gammay = gammay
        self.memory_capacity = memory_capacity
        self.target_update_steps = target_update_steps
        self.eval_net = net(*net_args).to(device)
        self.target_net = net(*net_args).to(device)
        self.memory = Memory(memory_capacity)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr)
        self.criterion = nn.SmoothL1Loss()

    @property
    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step_count / self.eps_decay)

    def choose_action(self, state: np.ndarray):
        if random.random() < self.eps:
            action = np.random.randint(0, self.n_actions)
        else:
            x = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(self.eval_net(x), 1).item()

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


class AgentBase:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.criterion = torch.nn.SmoothL1Loss()
        self.state = None
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None

    def init(self, net_dim, state_dim, action_dim):
        pass

    def select_action(self, state) -> np.ndarray:
        pass  # return action

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


class AgentDQN2(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim):  # explict call self.init() for multiprocessing
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.cri  # to keep the same from Actor-Critic framework

        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            action = self.act(states)[0]
            a_int = action.argmax().cpu().numpy()
        return a_int

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> Tuple[float, float]:
        buffer.update_now_len_before_sample()

        q_value = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return q_value.mean().item(), obj_critic.item()

    def get_obj_critic(self, buffer, batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.type(torch.long))
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value
