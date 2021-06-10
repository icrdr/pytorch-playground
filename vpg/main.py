from time import clock_getres
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch.nn.modules import activation
from torch.nn import MSELoss
from torch.optim import Adam
import gym
import math
from skimage.transform import resize

# discrete
# Stochastic policy
lr = 0.001
max_time = 200
env = gym.make('LunarLander-v2').unwrapped
action_n = env.action_space.n
state_n = env.observation_space.shape[0]
height = 20
width = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('workon', device)


class Mlp(nn.Module):
    def __init__(self, state_n, action_n, mid_n) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, action_n))

    def forward(self, x):
        return self.net(x)


def conv2d_size_out(size, kernel_size=3, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class Conv(nn.Module):
    def __init__(self, w, h, action_n, mid_n) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, mid_n, 3, 2), nn.BatchNorm2d(mid_n), nn.ReLU(),
                                  nn.Conv2d(mid_n, mid_n, 3, 2), nn.BatchNorm2d(mid_n), nn.ReLU())

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * mid_n
        self.head = nn.Linear(linear_input_size, action_n)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


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
        self.pi_net = Conv(width, height, action_n, 32).to(device)
        self.v_net = Conv(width, height, 1, 32).to(device)
        self.pi_optim = Adam(self.pi_net.parameters(), lr=lr)
        self.v_optim = Adam(self.v_net.parameters(), lr=lr)
        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        m = Categorical(logits=self.pi_net(state))
        action = m.sample().item()
        return action

    def learn(self):
        batch_state = torch.FloatTensor(self.states).to(device)
        batch_action = torch.FloatTensor(self.actions).to(device)
        batch_return = torch.FloatTensor(get_returns(self.rewards)).to(device)
        batch_reward_to_go = torch.FloatTensor(get_reward_to_go(self.rewards)).to(device)
        m = Categorical(logits=self.pi_net(batch_state))

        baseline = self.v_net(batch_state).detach()
        pi_loss = - (m.log_prob(batch_action) * (batch_reward_to_go - baseline)).mean()
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


def get_screen_state(env):

    screen = env.render(mode='rgb_array')
    screen = np.asarray(screen, dtype=np.float32) / 255
    screen = resize(screen, (height, width))
    current_screen = screen.transpose(2, 0, 1)

    global last_screen
    if(last_screen is None):
        last_screen = current_screen

    state = current_screen - last_screen
    last_screen = current_screen

    # state = current_screen
    return state


agent = Agent()
pi_loss_list = []
v_loss_list = []
reward_list = []
for episode_i in range(1000):
    state = env.reset()
    last_screen = None
    state = get_screen_state(env)
    done = False
    reward_total = 0
    for time_i in range(max_time):
        # env.render()
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = get_screen_state(env)
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
            if(episode_i % 30 == 0):
                print('episode {}: pi_loss {} v_loss {} reward {}'.format(
                    episode_i,
                    format(np.mean(pi_loss_list), '.5f'),
                    format(np.mean(v_loss_list), '.5f'),
                    format(np.mean(reward_list), '.2f')))
                loss_list = []
                reward_list = []

            break
