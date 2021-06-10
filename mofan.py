# %%
from agent import Memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from net import Net

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.memory = Memory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, *args):
        # transition = np.hstack((s, [a, r], s_))
        # # replace the old memory with new memory
        # index = self.memory_counter % MEMORY_CAPACITY
        # self.memory[index, :] = transition
        self.memory_counter += 1
        self.memory.add(*args)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        batch = self.memory.sample(BATCH_SIZE)
        final_batch = torch.BoolTensor(batch.final)
        next_state_batch = torch.FloatTensor(batch.next_state)  # shape (batch, 3, w, h)
        state_batch = torch.FloatTensor(batch.state)  # shape (batch, 3, w, h)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)  # shape (batch, 1)
        reward_batch = torch.FloatTensor(batch.reward)  # shape (batch, 1)

        # q_eval w.r.t the action in experience
        q = self.eval_net(state_batch).gather(1, action_batch)  # shape (batch, 1)
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        next_q[final_batch] = 0
        target_q = reward_batch + next_q * GAMMA
        target_q = target_q.unsqueeze(1)  # shape (batch, 1)
        loss = self.loss_func(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(s, a, s_, r, done)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_

# %%
