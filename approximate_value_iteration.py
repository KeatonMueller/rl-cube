import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy
from math import ceil

import cube as C
from cube_net import ResCubeNet
from data_generation import generate_training_data_avi

# see 10,000,000 unique states before updating parameters
STATES_PER_UPDATE = 10000000

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class AVI:
    '''
        class to implement approximate value iteration
    '''
    def __init__(self):
        self.model_train = ResCubeNet().to(device)
        self.model_label = ResCubeNet().to(device)
        self.model_label.load_state_dict(self.model_train.state_dict())

        self.optim_train = optim.Adam(self.model_train.parameters(), lr=0.01)
        self.optim_label = optim.Adam(self.model_label.parameters(), lr=0.01)
        self.optim_label.load_state_dict(self.optim_train.state_dict())

        self.seen_states = set()

    def cubes_to_input(self, samples):
        '''
            converts a list of Cube objects into a pytorch tensor for input into a network

            samples: list of Cube objects that are training examples
        '''
        # empty tensor to store input
        X = torch.empty(len(samples), 480)
        # for each cube
        for i, cube in enumerate(samples):
            # mark this state as seen
            self.seen_states.add(cube)
            # convert it to a tensor
            X[i] = cube.to_tensor()
        return X

    def label_samples(self, samples):
        '''
            label the given training examples

            samples: list of Cube objects that are training examples
        '''
        # empty tensor to store labels
        Y = torch.empty(len(samples), 1)
        # number of samples to put through the network at a time
        label_batch_size = 100
        # number of batches
        num_batches = ceil(len(samples) / label_batch_size)
        # empty tensor to store next states - input to network
        next_states = torch.empty(label_batch_size * 12, 480, device=device)
        self.model_label.eval()
        with torch.no_grad():
            for b in range(num_batches):
                # get the batch
                batch = samples[b * label_batch_size : (b + 1) * label_batch_size]
                num = 0
                # for each cube in the batch
                for cube in batch:
                    # make all 12 possible moves
                    for idx in range(12):
                        cube.idx_turn(idx)
                        # and store the state in next_states
                        next_states[num] = cube.to_tensor()
                        cube.idx_turn(idx, True)
                        num += 1
                # calculate the outputs for all next_states
                outputs = self.model_label(next_states)
                # for each cube in the batch
                for x in range(len(batch)):
                    # calculate the min cost from the outputs
                    min_cost = float('inf')
                    for i in range(x * 12, x * 12 + 12):
                        cost = 1 + outputs[i].item()
                        if cost < min_cost:
                            min_cost = cost
                    # store the min cost as the label for cube x
                    Y[b * label_batch_size + x] = min_cost
        # return the labels
        return Y

    def get_batches(self, X, Y, batch_size):
        '''
            convert given input and labels into batches

            X: training examples
            Y: labels
            batch_size: size of batches to create
        '''
        # randomly shuffle the input and output
        indices = torch.randperm(X.size()[0])
        X = X[indices]
        Y = Y[indices]

        # calculate number of batches
        num_batches = int(len(X) / batch_size)
        # empty tensors for batched training examples and labels
        X_batches = torch.empty(num_batches, batch_size, 480)
        Y_batches = torch.empty(num_batches, batch_size, 1)

        for b in range(num_batches):
            x_batch = X[b * batch_size : (b + 1) * batch_size]
            y_batch = Y[b * batch_size : (b + 1) * batch_size]

            X_batches[b] = x_batch
            Y_batches[b] = y_batch

        return X_batches, Y_batches

    def fit(self, batches):
        pass

    def train(self, num_scrambles, batch_size, max_updates):
        # check that the number of scrambles can be split into batches evenly
        # (this is up to the caller to ensure)
        if(num_scrambles % batch_size != 0):
            print(f'number of scrambles ({num_scrambles}) and batch size ({batch_size}) aren\'t compatible')
            exit()

        num_updates = 0
        for i in range(5): # this should be a while(num_updates < max_updates) once things are working
            samples = generate_training_data_avi(num_scrambles, 30)

            X = self.cubes_to_input(samples)
            Y = self.label_samples(samples)
            batches = self.get_batches(X, Y, batch_size)

            self.fit(batches)
            if(len(self.seen_states) > STATES_PER_UPDATE):
                self.model_label.load_state_dict(model_train.state_dict())
                self.optim_label.load_state_dict(optim_train.state_dict())
                num_updates += 1
