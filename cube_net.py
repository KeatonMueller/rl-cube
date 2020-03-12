import torch.nn as nn
from torch import tanh
from torch.nn.functional import softmax, leaky_relu, relu

class CubeNet(nn.Module):
    '''
        a CubeNet network used for the autodidactic iteration approach
        to solving a Rubik's Cube via reinforcement learning
    '''
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(20*24, 4096)     # input is a flattened 20*24 matrix
        self.h1 = nn.Linear(4096, 2048)         # first hidden layer

        # value branch
        self.h2_v = nn.Linear(2048, 512)
        self.out_v = nn.Linear(512, 1)

        # policy branch
        self.h2_p = nn.Linear(2048, 512)
        self.out_p = nn.Linear(512, 12)

        # initialize all weights with Glorot initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.h1.weight)
        nn.init.xavier_uniform_(self.h2_v.weight)
        nn.init.xavier_uniform_(self.h2_p.weight)
        nn.init.xavier_uniform_(self.out_v.weight)
        nn.init.xavier_uniform_(self.out_p.weight)

    def forward(self, x):
        x = leaky_relu(self.input(x))
        x = leaky_relu(self.h1(x))

        # value branch
        x_v = leaky_relu(self.h2_v(x))
        out_v = leaky_relu(self.out_v(x_v))

        # policy branch
        x_p = tanh(self.h2_p(x))
        out_p = softmax(self.out_p(x_p), dim=1)

        return (out_v, out_p)

class ResCubeNet(nn.Module):
    '''
        a Residual CubeNet network used for the value iteration approach
        to solving a Rubik's Cube via reinforcement learning
    '''
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20*24, 5000)        # first fully connected layer
        self.bn1 = nn.BatchNorm1d(5000)          # batch normalization layer
        self.fc2 = nn.Linear(5000, 1000)         # second fully connected layer
        self.bn2 = nn.BatchNorm1d(1000)          # batch normalization layer

        # layers for first residual block
        self.res1a = nn.Linear(1000, 1000)
        self.res1a_bn = nn.BatchNorm1d(1000)
        self.res1b = nn.Linear(1000, 1000)
        self.res1b_bn = nn.BatchNorm1d(1000)

        # layers for second residual block
        self.res2a = nn.Linear(1000, 1000)
        self.res2a_bn = nn.BatchNorm1d(1000)
        self.res2b = nn.Linear(1000, 1000)
        self.res2b_bn = nn.BatchNorm1d(1000)

        # layers for third residual block
        self.res3a = nn.Linear(1000, 1000)
        self.res3a_bn = nn.BatchNorm1d(1000)
        self.res3b = nn.Linear(1000, 1000)
        self.res3b_bn = nn.BatchNorm1d(1000)

        # layers for fourth residual block
        self.res4a = nn.Linear(1000, 1000)
        self.res4a_bn = nn.BatchNorm1d(1000)
        self.res4b = nn.Linear(1000, 1000)
        self.res4b_bn = nn.BatchNorm1d(1000)

        # output layer
        self.out = nn.Linear(1000, 1)

        # identity layer for residual connections
        self.identity = nn.Identity(1000)

    def forward(self, x):
        # first two hidden layers
        x = relu(self.bn1(self.fc1(x)))
        x = relu(self.bn2(self.fc2(x)))

        # first residual block
        residual = self.identity(x)
        x = relu(self.res1a_bn(self.res1a(x)))
        x = relu(self.res1b_bn(self.res1b(x)) + residual)

        # second residual block
        residual = self.identity(x)
        x = relu(self.res2a_bn(self.res2a(x)))
        x = relu(self.res2b_bn(self.res2b(x)) + residual)

        # third residual block
        residual = self.identity(x)
        x = relu(self.res3a_bn(self.res3a(x)))
        x = relu(self.res3b_bn(self.res3b(x)) + residual)

        # fourth residual block
        residual = self.identity(x)
        x = relu(self.res4a_bn(self.res4a(x)))
        x = relu(self.res4b_bn(self.res4b(x)) + residual)

        return self.out(x)
