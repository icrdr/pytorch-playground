
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
# CartPole-v1
# LunarLander-v2
# Assault-v0
# Breakout-v0

lr = 1e-4
max_time = 300
env = gym.make('LunarLander-v2').unwrapped
action_n = env.action_space.n
state_n = env.observation_space.shape[0]
episode_num = 100000
iter_num = 10
clip_ratio = 0.2
height = 100
width = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
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


class Buffer:
    def __init__(self) -> None:
        pass


class Agent:
    def __init__(self) -> None:
        self.pi_net = Mlp(state_n, action_n, 64).to(device)
        self.v_net = Mlp(state_n, 1, 32).to(device)
        # self.pi_net = Conv(width, height, action_n, 32).to(device)
        # self.v_net = Conv(width, height, 1, 32).to(device)
        self.pi_optim = Adam(self.pi_net.parameters(), lr=lr)
        self.v_optim = Adam(self.v_net.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        m = Categorical(logits=self.pi_net(state))

        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.item()

    def learn(self, states, actions, rewards_to_go, old_log_probs):
        batch_state = torch.FloatTensor(states).to(device)
        batch_action = torch.FloatTensor(actions).to(device)
        batch_reward_to_go = torch.FloatTensor(rewards_to_go).to(device)
        batch_old_logps = torch.FloatTensor(old_log_probs).to(device)

        m = Categorical(logits=self.pi_net(batch_state))

        ratio = torch.exp(m.log_prob(batch_action) - batch_old_logps)
        clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        adv = batch_reward_to_go - self.v_net(batch_state).squeeze(1).detach()

        pi_loss = - (torch.min(ratio * adv, clip * adv)).mean()
        v_loss = MSELoss()(self.v_net(batch_state).squeeze(1), batch_reward_to_go)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        return pi_loss.item(), v_loss.item()


# def get_screen_state(env):

#     screen = env.render(mode='rgb_array')
#     screen = np.asarray(screen, dtype=np.float32) / 255
#     screen = resize(screen, (height, width))
#     current_screen = screen.transpose(2, 0, 1)

#     global last_screen
#     if(last_screen is None):
#         last_screen = current_screen

#     state = current_screen - last_screen
#     last_screen = current_screen

#     # state = current_screen
#     return state


agent = Agent()
pi_loss_list = []
v_loss_list = []
return_list = []
for episode_i in range(episode_num):
    states = []
    actions = []
    rewards = []
    log_probs = []
    state = env.reset()
    # last_screen = None
    # state = get_screen_state(env)
    _return = 0
    for time_i in range(max_time):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # next_state = get_screen_state(env)
        if(time_i == max_time - 1):
            done = True
            reward = -100

        if(episode_i % 500 == 0):
            env.render()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        _return += reward

        if done:
            rewards_to_go = get_reward_to_go(rewards)
            for i in range(iter_num):
                pi_loss, v_loss = agent.learn(states, actions, rewards_to_go, log_probs)

            pi_loss_list.append(pi_loss)
            v_loss_list.append(v_loss)
            return_list.append(_return)
            if(episode_i % 50 == 0):
                print('episode {}: pi_loss {} v_loss {} return {}'.format(
                    episode_i,
                    format(np.mean(pi_loss_list), '.3f'),
                    format(np.mean(v_loss_list), '.3f'),
                    format(np.mean(return_list), '.2f')))
                loss_list = []
                return_list = []
            break
