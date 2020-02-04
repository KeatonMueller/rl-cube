import torch.nn as nn
import torch.nn.functional as F

class CubeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(20*24, 4096)     # input is a flattened 20*24 matrix
        self.h1 = nn.Linear(4096, 2048)         # first hidden layer

        # value branch
        self.h2_v = nn.Linear(2048, 512)
        self.output_v = nn.Linear(512, 1)

        # policy branch
        self.h2_p = nn.Linear(2048, 512)
        self.output_p = nn.Linear(512, 12)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.h1(x))

        # value branch
        x_v = F.relu(self.h2_v(x))
        x_v = F.log_softmax(self.output_v(x_v), dim=1)

        # policy branch
        x_p = F.relu(self.h2_p(x))
        x_p = F.relu(self.output_p(x_p))

        return (x_v, x_p)
