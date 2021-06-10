from mc import MLPActorCritic
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


class Mlp(nn.Module):
    def __init__(self, state_n, action_n, mid_n) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, mid_n), nn.ReLU(),
                                 nn.Linear(mid_n, action_n), nn.Identity())

    def forward(self, x):
        return self.net(x)


def conv2d_size_out(size, kernel_size=3, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


def get_reward_to_go(rewards, gamma=0.99):
    n = len(rewards)
    rewards_to_go = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        rewards_to_go[i] = gamma * rewards[i] + (rewards_to_go[i + 1] if i + 1 < n else 0)
    return rewards_to_go


class Agent:
    def __init__(self) -> None:
        self.pi_net = Mlp(state_n, action_n, 64).to(device)
        self.v_net = Mlp(state_n, 1, 64).to(device)
        self.pi_optim = Adam(self.pi_net.parameters(), lr=pi_lr)
        self.v_optim = Adam(self.v_net.parameters(), lr=v_lr)
        log_std = -0.5 * np.ones(action_n, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            std = torch.exp(self.log_std)
            n = Normal(self.pi_net(state), std)

            action = n.sample()
            log_prob = n.log_prob(action).sum(-1)
            value = self.v_net(state)

        return action.numpy(), value.numpy(), log_prob.numpy()

    def learn(self, data):
        batch_state, batch_action, adv, ret, val, batch_old_logps = data[
            'obs'], data['act'], data['adv'], data['ret'], data['val'], data['logp']

        std = torch.exp(self.log_std)
        n = Normal(self.pi_net(batch_state), std)

        ratio = torch.exp(n.log_prob(batch_action).sum(-1) - batch_old_logps)
        clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        adv = ret - self.v_net(batch_state).squeeze(-1).detach()
        pi_loss = -(torch.min(ratio * adv, clip * adv)).mean()
        v_loss = MSELoss()(self.v_net(batch_state).squeeze(-1), ret)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        return pi_loss.item(), v_loss.item()


class Buffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim))
        self.act_buf = np.zeros((size, act_dim))
        self.adv_buf = np.zeros(size)
        self.rew_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.val_buf = np.zeros(size)
        self.logp_buf = np.zeros(size)
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew, val, logp):

        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + 0.99 * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = get_reward_to_go(deltas, 0.9)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = get_reward_to_go(rews)[:-1]

        self.path_start_idx = self.ptr

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

    def get(self):
        assert self.ptr == self.max_size   # buffer has to be full before you can get

        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, val=self.val_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


agent = Agent()
buf = Buffer(state_n, action_n, max_time)
pi_loss_list = []
v_loss_list = []
return_list = []

for episode_i in range(episode_num):
    state = env.reset()
    _return = 0
    for time_i in range(max_time):
        action, value, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        if(time_i == max_time - 1):
            done = True
            # reward = -100

        # if(episode_i % 100 == 0):
        #     env.render()

        buf.store(state, action, reward, value, log_prob)

        state = next_state
        _return += reward

        if done:
            buf.finish_path()
            state = env.reset()
            return_list.append(_return)
            _return = 0

    data = buf.get()
    for i in range(iter_num):
        pi_loss, v_loss = agent.learn(data)

        pi_loss_list.append(pi_loss)
        v_loss_list.append(v_loss)

    buf.reset()

    if(episode_i % 1 == 0):
        print('episode {}: pi_loss {} v_loss {} return {}'.format(
            episode_i,
            format(np.mean(pi_loss_list), '.3f'),
            format(np.mean(v_loss_list), '.3f'),
            format(np.mean(return_list), '.2f')))
        pi_loss_list = []
        v_loss_list = []
        return_list = []
