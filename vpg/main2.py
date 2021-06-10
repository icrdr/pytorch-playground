from time import clock_getres
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
from torch.nn.modules import activation
from torch.nn import MSELoss
from torch.optim import Adam
import gym
import math

# continuous
# Stochastic policy
lr = 0.01
max_time = 1000
env = gym.make('LunarLanderContinuous-v2').unwrapped
action_n = env.action_space.shape[0]
state_n = env.observation_space.shape[0]


class Mlp(nn.Module):
    def __init__(self, state_n, action_n, mid_n) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, action_n))

    def forward(self, x):
        return self.net(x)

# class Conv(nn.Module):
#     def __init__(self, state_n, action_n, mid_n) -> None:
#         super().__init__()
#         self.net = nn.Sequential(nn.Conv2d(state_n, mid_n), nn.ReLU(),
#                                  nn.Linear(mid_n, mid_n), nn.ReLU(),
#                                  nn.Linear(mid_n, mid_n), nn.ReLU(),
#                                  nn.Linear(mid_n, action_n))

#     def forward(self, x):
#         return self.net(x)


def get_reward_to_go(rewards, gamma=0.99):
    n = len(rewards)
    rewards_to_go = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        rewards_to_go[i] = gamma * rewards[i] + (rewards_to_go[i + 1] if i + 1 < n else 0)
    return rewards_to_go


def get_returns(rewards):
    _return = 0
    for i in range(len(rewards)):
        _return += rewards[i]
    return [_return] * len(rewards)


class Agent:
    def __init__(self) -> None:
        self.pi_net = Mlp(state_n, action_n, 64)
        self.v_net = Mlp(state_n, 1, 64)
        self.pi_optim = Adam(self.pi_net.parameters(), lr=lr)
        self.v_optim = Adam(self.v_net.parameters(), lr=lr)
        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state.copy())
        n = Normal(self.pi_net(state), 1)
        action = n.sample().numpy()
        return action

    def learn(self):
        batch_state = torch.FloatTensor(self.states)
        batch_action = torch.FloatTensor(self.actions)
        batch_return = torch.FloatTensor(get_returns(self.rewards))
        batch_reward_to_go = torch.FloatTensor(get_reward_to_go(self.rewards))
        n = Normal(self.pi_net(batch_state), 1)

        baseline = self.v_net(batch_state).detach()
        pi_loss = - (n.log_prob(batch_action).sum(1) * (batch_reward_to_go - baseline)).mean()
        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        v_loss = MSELoss()(self.v_net(batch_state).squeeze(1), batch_reward_to_go)
        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        self.states = []
        self.actions = []
        self.rewards = []

        return pi_loss.item(), v_loss.item()

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


agent = Agent()
pi_loss_list = []
v_loss_list = []
reward_list = []
for episode_i in range(1000):
    state = env.reset()
    done = False
    reward_total = 0
    for time_i in range(max_time):
        env.render()
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)

        if(time_i == max_time - 1):
            done = True
            reward = -100

        agent.store(state, action, reward)

        state = next_state
        reward_total += reward

        if done:
            pi_loss, v_loss = agent.learn()
            pi_loss_list.append(pi_loss)
            v_loss_list.append(v_loss)
            reward_list.append(reward_total)
            if(episode_i % 20 == 0):
                print('episode {}: pi_loss {} v_loss {} reward {}'.format(
                    episode_i,
                    format(np.mean(pi_loss_list), '.5f'),
                    format(np.mean(v_loss_list), '.5f'),
                    format(np.mean(reward_list), '.2f')))
                loss_list = []
                reward_list = []

            break
