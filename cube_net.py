import torch.nn as nn
from torch import tanh
from torch.nn.functional import softmax

class CubeNet(nn.Module):
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
