import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
from torch.nn.modules import activation
from torch.nn import MSELoss
from torch.optim import Adam
import gym
import math
from skimage.transform import resize
from copy import deepcopy

# discrete
# Stochastic policy
# CartPole-v1
# LunarLander-v2
# Assault-v0
# Breakout-v0
# BipedalWalker-v3
pi_lr = 3e-4
v_lr = 1e-2
max_time = 4000
env = gym.make('LunarLanderContinuous-v2').unwrapped
action_n = env.action_space.shape[0]
state_n = env.observation_space.shape[0]
episode_num = 1000
iter_num = 80
clip_ratio = 0.2
height = 100
width = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
target_kl = 0.01
print('workon', device)
gamma = 0.99


class Buffer:
    def __init__(self, state_n, action_n, capactiy):
        self.states = np.zeros((capactiy, state_n))
        self.next_states = np.zeros((capactiy, state_n))
        self.actions = np.zeros((capactiy, action_n))
        self.rewards = np.zeros(capactiy)
        self.dones = np.zeros(capactiy)
        self.capactiy = capactiy
        self.current_index = 0
        self.current_size = 0

    def store(self, state, next_state, action, reward, done):
        self.states[self.current_index] = state
        self.next_states[self.current_index] = next_state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.dones[self.current_index] = done
        self.current_index = (self.current_index + 1) % self.capactiy
        self.current_size = min(self.current_size + 1, self.capactiy)

    def __len__(self):
        return len(self.memory)

    def batch(self, batch_size=32):
        indexs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(states=self.states[indexs],
                     next_states=self.next_states[indexs],
                     actions=self.actions[indexs],
                     rewards=self.rewards[indexs],
                     dones=self.dones[indexs])
        return batch


class Mlp(nn.Module):
    def __init__(self, state_n, action_n, mid_n) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, action_n), nn.Identity())

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self) -> None:
        self.pi = Mlp(state_n, action_n, 64)
        self.q = Mlp(state_n + action_n, 1, 64)

        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)

        self.pi_optim = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optim = Adam(self.q.parameters(), lr=v_lr)

    def step(self, state, noise_scale):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.pi(state)
            action += noise_scale * np.random.randn(action_n)
        return action.numpy()

    def learn(self, batch):
        q_value = self.q(batch.states, batch.actions)
        q_next_value = self.q_target(batch.next_states, self.pi(batch.next_states)).detach()
        q_target_value = batch.rewards + gamma * (1 - batch.dones) * q_next_value

        q_loss = MSELoss()(q_target_value, q_value)
        pi_loss = -self.q(batch.next_states, self.pi(batch.next_states)).mean()

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        return pi_loss.item(), q_loss.item()


agent = Agent()
buffer = Buffer(state_n, action_n, max_time)
pi_loss_list = []
q_loss_list = []
return_list = []

for episode_i in range(episode_num):
    state = env.reset()
    return_ = 0
    for time_i in range(max_time):
        action, value, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        if(time_i == max_time - 1):
            done = True
            # reward = -100

        # if(episode_i % 100 == 0):
        #     env.render()

        buffer.store(state, next_state, action, reward, done)

        state = next_state
        return_ += reward

        if done:
            state = env.reset()
            return_list.append(return_)
            return_ = 0

    for i in range(iter_num):
        pi_loss, q_loss = agent.learn(buffer.sample)

        pi_loss_list.append(pi_loss)
        q_loss_list.append(q_loss)

    if(episode_i % 1 == 0):
        print('episode {}: pi_loss {} v_loss {} return {}'.format(
            episode_i,
            format(np.mean(pi_loss_list), '.3f'),
            format(np.mean(q_loss_list), '.3f'),
            format(np.mean(return_list), '.2f')))
        pi_loss_list = []
        v_loss_list = []
        return_list = []
