import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=3, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class ConvNet(nn.Module):
    def __init__(self, w, h, action_n):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.bn3 = nn.BatchNorm2d(64)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, action_n)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x


class Net(nn.Module):
    def __init__(self, state_n, action_n):
        super(Net, self).__init__()
        self.l1 = nn.Linear(state_n, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, action_n)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x
