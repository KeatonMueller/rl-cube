import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy

import cube as C
from cube_net import ResCubeNet

# see 10,000,000 unique states before updating parameters
STATES_PER_UPDATE = 10000000

class AVI:
    '''
        class to implement approximate value iteration
    '''
    def __init__(self):
        self.model_train = ResCubeNet()
        self.model_label = ResCubeNet()
        self.model_label.load_state_dict(model_train.state_dict())

        self.optim_train = optim.Adam(self.model_train.parameters(), lr=0.01)
        self.optim_label = optim.Adam(self.model_label.parameters(), lr=0.01)
        self.optim_label.load_state_dict(optim_train.state_dict())

        self.seen_states = set()

    def get_batches(self, X, batch_size):
        self.model_label.eval()
        with torch.no_grad:
            # empty tensor of labels
            Y = torch.empty(len(X))
            # for each training example
            for i, cube in enumerate(X):
                # find min cost over 12 possible moves
                min_cost = float('inf')
                for idx in range(12):
                    # perform a move
                    cube.idx_turn(idx)
                    # calculate cost
                    cost = 1 + self.model_label(cube.to_tensor())
                    # update min
                    if cost < min_cost:
                        min_cost = cost
                    # undo the move
                    cube.idx_turn(idx, True)
                # record this label
                Y[i] = min_cost
        # return labels
        return Y

    def fit(self, X, Y):
        pass

    def train(self, iterations, num_scrambles, batch_size, max_updates):
        num_updates = 0
        for i in iterations:
            X = generate_training_data_avi(num_scrambles, 30)
            batches = self.get_batches(X, batch_size)
            fit(X, Y)
            if(len(seen_states) > STATES_PER_UPDATE):
                self.model_label.load_state_dict(model_train.state_dict())
                self.optim_label.load_state_dict(optim_train.state_dict())
                num_updates += 1
                if(num_updates > max_updates):
                    break
