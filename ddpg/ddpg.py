from core import MLPActorCritic
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


# BipedalWalker-v3
pi_lr = 1e-3
qf_lr = 1e-3
# LunarLanderContinuous-v2
env = gym.make('BipedalWalker-v3').unwrapped
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
act_limit = env.action_space.high[0]
episode_steps_num = 4000
episode_iters_num = 1000
max_steps_per_game = 500
train_iters_num = 50
clip_ratio = 0.2
height = 100
width = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
target_kl = 0.01
print('workon', device)
gamma = 0.99
polyak = 0.99
update_after = 1000
print(act_limit)
print(state_dim)
print(action_dim)


class Buffer:
    def __init__(self, state_dim, action_dim, capactiy):
        self.states = np.zeros((capactiy, state_dim))
        self.next_states = np.zeros((capactiy, state_dim))
        self.actions = np.zeros((capactiy, action_dim))
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

    def batch(self, batch_size=128):
        assert batch_size <= self.current_size
        indexs = np.random.randint(0, self.current_size, size=batch_size)
        batch = {'states': self.states[indexs],
                 'next_states': self.next_states[indexs],
                 'actions': self.actions[indexs],
                 'rewards': self.rewards[indexs],
                 'dones': self.dones[indexs]}

        return batch


class Mlp(nn.Module):
    def __init__(self, state_dim, action_dim, mid_n, out_activation=nn.Identity) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, action_dim), out_activation())

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, state_dim, action_dim) -> None:
        self.pi = Mlp(state_dim, action_dim, 64, nn.Tanh)
        self.qf = Mlp(state_dim + action_dim, 1, 64)

        self.pi_target = deepcopy(self.pi)
        self.qf_target = deepcopy(self.qf)
        # for p in self.pi_target.parameters():
        #     p.requires_grad = False
        # for p in self.qf_target.parameters():
        #     p.requires_grad = False
        self.pi_optim = Adam(self.pi.parameters(), lr=pi_lr)
        self.qf_optim = Adam(self.qf.parameters(), lr=qf_lr)

        # # Create actor-critic module and target networks
        # self.ac = MLPActorCritic(env.observation_space, env.action_space)
        # self.ac_targ = deepcopy(self.ac)

        # # Freeze target networks with respect to optimizers (only update via polyak averaging)
        # for p in self.ac_targ.parameters():
        #     p.requires_grad = False

        # self.pi_optim = Adam(self.ac.pi.parameters(), lr=pi_lr)
        # self.qf_optim = Adam(self.ac.q.parameters(), lr=qf_lr)

    def step(self, state, noise_scale):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = act_limit * self.pi(state).numpy()
            action += noise_scale * np.random.randn(action_dim)

        # action = self.ac.act(torch.as_tensor(state, dtype=torch.float32))
        # action += noise_scale * np.random.randn(action_dim)
        return np.clip(action, -act_limit, act_limit)

    def learn(self, batch):
        states = torch.FloatTensor(batch['states'])
        next_states = torch.FloatTensor(batch['next_states'])
        actions = torch.FloatTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        dones = torch.BoolTensor(batch['dones'])

        q_value = self.qf(torch.cat([states, actions], dim=-1))
        with torch.no_grad():
            q_next_value = self.qf_target(torch.cat([next_states, self.pi_target(next_states)], dim=-1))

        q_next_value[dones] = 0
        q_target_value = rewards.unsqueeze(-1) + gamma * q_next_value
        qf_loss = MSELoss()(q_target_value, q_value)

        self.qf_optim.zero_grad()
        qf_loss.backward()
        self.qf_optim.step()

        # frezee qf param
        for param in self.qf.parameters():
            param.requires_grad = False

        pi_loss = -self.qf(torch.cat([next_states, self.pi(next_states)], dim=-1)).mean()
        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        for param in self.qf.parameters():
            param.requires_grad = True

        with torch.no_grad():
            for param, param_target in zip(self.qf.parameters(), self.qf_target.parameters()):
                param_target.data.mul_(polyak)
                param_target.data.add_((1 - polyak) * param.data)

            for param, param_target in zip(self.pi.parameters(), self.pi_target.parameters()):
                param_target.data.mul_(polyak)
                param_target.data.add_((1 - polyak) * param.data)

        # self.qf_target.load_state_dict(self.qf.state_dict())
        # self.pi_target.load_state_dict(self.pi.state_dict())

        return pi_loss.item(), qf_loss.item()


agent = Agent(state_dim, action_dim)
buffer = Buffer(state_dim, action_dim, int(1e6))
pi_loss_list = []
qf_loss_list = []
return_list = []

for episode_i in range(episode_iters_num):
    state = env.reset()
    total_reward = 0
    step_index = 0
    for step_i in range(episode_steps_num):
        action = agent.step(state, 0.3)
        next_state, reward, done, _ = env.step(action)

        if(step_index == max_steps_per_game - 1):
            done = True
            # reward = -100

        buffer.store(state, next_state, action, reward, done)

        state = next_state
        total_reward += reward
        step_index += 1

        if done:
            state = env.reset()
            return_list.append(total_reward)
            total_reward = 0
            step_index = 0
            
        if step_i >= update_after and step_i % train_iters_num == 0:
            for i in range(train_iters_num):
                pi_loss, qf_loss = agent.learn(buffer.batch())
                pi_loss_list.append(pi_loss)
                qf_loss_list.append(qf_loss)

    if(episode_i % 40 == 0 and episode_i != 0):
        state = env.reset()
        total_reward = 0
        for step_i in range(max_steps_per_game):
            action = agent.step(state, 0)
            state, reward, done, _ = env.step(action)
            env.render()
            if(step_index == max_steps_per_game - 1):
                done = True
            total_reward += reward
            if done:
                print('test | return: {}'.format(total_reward))
                break

    if(episode_i % 1 == 0):
        print('episode {}| pi_loss {} qf_loss {} return {}'.format(
            episode_i,
            format(np.mean(pi_loss_list), '.3f'),
            format(np.mean(qf_loss_list), '.3f'),
            format(np.mean(return_list), '.2f')))
        pi_loss_list = []
        qf_loss_list = []
        return_list = []
