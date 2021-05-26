# %%
from warnings import formatwarning
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
from torch.types import Number
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, w, h, action_n):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_n)

    def forward(self, x: Tensor):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, capacity: int):
        # when new items are added, a corresponding number
        # of items are discarded from the opposite end.
        self.memory = deque([], maxlen=capacity)

    def add(self, state: list, action: list, next_state: list, reward: list):
        self.memory.append(Transition(state, action, next_state, reward))

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
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self, w, h, n_actions):
        self.n_actions = n_actions
        self.step_count = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.eval_net = Net(w, h, n_actions).to(device)
        self.target_net = Net(w, h, n_actions).to(device)
        self.memory = Memory(1000)

    def choose_action(self, state: np.ndarray):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.step_count / self.eps_decay)
        self.step_count += 1

        if random.random() < eps_threshold:
            x = torch.unsqueeze(torch.from_numpy(state), 0)
            q_values = self.eval_net.forward(x)
            action = torch.max(q_values, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).numpy()


env = gym.make('CartPole-v0')
num_episodes = 20
for i_episode in range(num_episodes):
    state = env.reset()
    state = np.expand_dims(np.asarray(state), axis=0)
    state = np.expand_dims(state, axis=0)
    print(state)
    agent = Agent(1, 4, env.action_space.n)
    for t in range(1000):
        env.render()
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
