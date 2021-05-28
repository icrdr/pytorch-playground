# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, state_dims, fc1_dims, fc2_dims, action_dims):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    def __init__(self, state_dims, action_dims,
                 memo_size=100000, batch_size=64,
                 epsilon=1.0, gamma=0.99,
                 eps_des=5e-4, eps_end=0.01):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.memo_size = memo_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.eps_des = eps_des
        self.eps_end = eps_end
        self.cntr = 0
        self.Q = DQN(state_dims, 256, 256, action_dims)
        self.state_memo = np.zeros(
            (memo_size, state_dims), dtype=np.float32)
        self.new_state_memo = np.zeros(
            (memo_size, state_dims), dtype=np.float32)
        self.action_memo = np.zeros(memo_size, dtype=np.int32)
        self.reward_memo = np.zeros(memo_size, dtype=np.float32)
        self.done_memo = np.zeros(memo_size, dtype=np.bool)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def store_transition(self, state, new_state, action, reward, done):
        index = self.cntr % self.memo_size
        self.state_memo[index] = state
        self.new_state_memo[index] = new_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = done
        self.cntr += 1

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                actions = self.Q(state)
            return torch.argmax(actions).item()
        else:
            return np.random.choice(self.action_dims)

    def learn(self):
        if self.cntr < self.batch_size:
            return
        self.optimizer.zero_grad()

        max_memo = min(self.cntr, self.memo_size)
        batch = np.random.choice(max_memo, self.batch_size)
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_memo[batch])
        new_state_batch = torch.tensor(self.new_state_memo[batch])
        action_batch = torch.tensor(self.action_memo[batch], dtype=torch.int64)
        reward_batch = torch.tensor(self.reward_memo[batch])
        done_batch = torch.tensor(self.done_memo[batch])

        q_eval = self.Q(state_batch).gather(
            1, action_batch.unsqueeze(1)).squeeze(1)
        q_next = self.Q(new_state_batch)
        q_next[done_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next.max(dim=1)[0]
        loss = self.loss(q_target, q_eval)

        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_des)


# %%
env = gym.make('CartPole-v1')
agent = Agent(state_dims=env.observation_space.shape[0],
              action_dims=env.action_space.n,
              memo_size=100000,
              epsilon=1,
              batch_size=64)
n_epochs = 200
scores = []
for i in range(n_epochs):
    done = False
    score = 0
    state = env.reset()
    while not done:
        # action = env.action_space.sample()
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)
        agent.store_transition(state, new_state, action, reward, done)
        agent.learn()

        score += reward
        state = new_state
        # env.render()

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('epoch %d, score: %.2f, avg score: %.2f, eps: %.2f' %
          (i, score, avg_score, agent.epsilon))


# %%
for i in range(10):
    done = False
    score = 0
    state = env.reset()
    while not done:
        state = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            actions = agent.Q(state)
        action = torch.argmax(actions).item()
        new_state, reward, done, info = env.step(action)

        score += reward
        state = new_state
        env.render()
    print(score)

# %%
