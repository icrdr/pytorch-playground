# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from tqdm import tqdm
import wandb
import random


def conv_size_cal(w, kernel, stride, padding=0, dilation=1):
    return (w + 2*padding - dilation*(kernel-1) - 1) // stride + 1


def conv2d_size_cal(w, h, kernel, stride, padding=0, dilation=1):
    w = conv_size_cal(w, kernel, stride, padding, dilation)
    h = conv_size_cal(h, kernel, stride, padding, dilation)
    return w, h


class DQNConv(nn.Module):
    def __init__(self, w, h, action_dims):
        super(DQNConv, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 5, 2)
        w, h = conv2d_size_cal(w, h, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        w, h = conv2d_size_cal(w, h, 5, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        w, h = conv2d_size_cal(w, h, 3, 2)
        self.fc1 = nn.Linear(64*w*h, 256)
        self.fc2 = nn.Linear(256, action_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_dims, fc1_dims, fc2_dims, fc3_dims, action_dims):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.adv_fc1 = nn.Linear(fc2_dims, fc3_dims)
        self.adv_fc2 = nn.Linear(fc3_dims, action_dims)

        self.val_fc1 = nn.Linear(fc2_dims, fc3_dims)
        self.val_fc2 = nn.Linear(fc3_dims, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        adv = self.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        val = self.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        x = val + adv - adv.mean(1)
        return x


class Memory():
    def __init__(self, size):
        self.size = size
        self.transitions = []
        self.cntr = 0

    def store(self, *args):
        index = self.cntr % self.size
        transition = [*args]
        if len(self.transitions) < self.size:
            self.transitions.append(transition)
        else:
            self.transitions[index] = transition
        self.cntr += 1

    def sample(self, batch_size):
        assert len(self.transitions) >= batch_size, "Not enough transitions."
        batch = random.sample(self.transitions, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.transitions)


class Agent():
    def __init__(self, w, h, action_dims,
                 memory_size=100000, batch_size=64,
                 epsilon=1, gamma=0.99,
                 eps_des=1e-4, eps_end=0.01,
                 device=torch.device("cpu")):
        self.w = w
        self.h = h
        self.action_dims = action_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.eps_des = eps_des
        self.eps_end = eps_end
        self.device = device
        self.learn_cntr = 0
        self.last_screen = None

        self.Q = DQNConv(w, h, self.action_dims).to(self.device)
        self.target_Q = DQNConv(w, h, self.action_dims).to(self.device)
        self.target_Q.eval()

        self.memory = Memory(memory_size)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def choose_action(self, state):
        current_screen = np.mean(state[25:195, 30:120, :], axis=2)
        if self.last_screen is None:
            self.last_screen = current_screen
        state = np.stack([self.last_screen, current_screen])
        self.last_screen = current_screen
        if np.random.random() > self.epsilon:
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                actions = self.Q(state)
            return torch.argmax(actions).item()
        else:
            return np.random.choice(self.action_dims)

    def record(self, state, new_state, action, reward, done):
        current_screen = np.mean(state[25:195, 30:120, :], axis=2)
        next_screen = np.mean(new_state[25:195, 30:120, :], axis=2)
        if self.last_screen is None:
            self.last_screen = current_screen

        state = np.stack([self.last_screen, current_screen])
        new_state = np.stack([current_screen, next_screen])
        self.memory.store(state, new_state, action, reward, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(self.device)
        new_state_batch = torch.tensor(batch[1], dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[2], dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(batch[3], dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool).to(self.device)

        q_eval = self.Q(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_next = self.target_Q(new_state_batch)
        # # nature DQN
        # max_q_next = q_next.max(dim=1)[0]

        # DDQN
        max_action_batch = self.Q(new_state_batch).argmax(dim=1)
        max_q_next = q_next.gather(1, max_action_batch.unsqueeze(1)).squeeze(1)

        # mask last action
        max_q_next[done_batch] = 0.0

        q_target = reward_batch + self.gamma * max_q_next
        loss = self.loss(q_target, q_eval)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_des)

        if self.learn_cntr % 20 == 0:
            self.target_Q.load_state_dict(self.Q.state_dict())
        self.learn_cntr += 1

        return loss.item()


# %%
"""
SpaceInvaders-v0 [25:195, 30:120, :] 170 90
Breakout-v0 [35:195, 10:150, :] 160 140
Pong-v0: [33:195, 15:145, :] 160 130
MsPacman-v0: [:172, :, :] 172 160
"""
wandb.init(project="atari")
env = gym.make('SpaceInvaders-v0')

agent = Agent(170, 90,
              action_dims=env.action_space.n,
              memory_size=1000,
              batch_size=64,
              eps_des=5e-4)
wandb.watch(agent.Q)

n_epochs = 500
scores = []
for i in range(n_epochs):
    done = False
    score = 0
    state = env.reset()
    losses = []
    while not done:
        # action = env.action_space.sample()
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)
        agent.record(state, new_state, action, reward, done)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        score += reward
        state = new_state
        env.render()
    avg_loss = np.mean(losses)
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    wandb.log({"Loss": avg_loss, "Score": score})
    print('epoch %d, score: %.2f, avg score: %.2f, eps: %.2f' %
          (i, score, avg_score, agent.epsilon))


# %%
for i in range(10):
    done = False
    score = 0
    state = env.reset()
    while not done:
        current_screen = np.mean(state[25:195, 30:120, :], axis=2)
        if agent.last_screen is None:
            agent.last_screen = current_screen
        state = np.stack([agent.last_screen, current_screen])
        state = torch.tensor([state], dtype=torch.float32).to(agent.device)
        with torch.no_grad():
            actions = agent.Q(state)
        action = torch.argmax(actions).item()
        new_state, reward, done, info = env.step(action)

        score += reward
        state = new_state
        env.render()
    print(score)


# %%
