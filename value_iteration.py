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
        self.model_train = ResCubeNet()
        self.model_label = ResCubeNet()
        self.model_label.load_state_dict(self.model_train.state_dict())

        self.optim_train = optim.Adam(self.model_train.parameters(), lr=0.01)
        self.optim_label = optim.Adam(self.model_label.parameters(), lr=0.01)
        self.optim_label.load_state_dict(self.optim_train.state_dict())

        self.seen_states = set()

    def cubes_to_input(self, samples):
        '''
            converts a list of Cube objects into a pytorch tensor for input into a network
        '''
        pass

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

    def get_batches(self, examples, batch_size):
        '''
            label the given training examples and return them in batches

            examples: list of Cube objects that are training examples
            batch_size: size of batches to create
        '''
        # calculate number of batches
        num_batches = int(len(examples) / batch_size)
        # empty tensors for training examples and labels
        X = torch.empty(num_batches, batch_size, 480)
        Y = torch.empty(num_batches, batch_size, 1)
        self.model_label.eval()
        with torch.no_grad():
            # for each batch of training examples
            for b in range(num_batches):
                # get batch of training examples in array form
                training_batch = examples[b * batch_size : (b + 1) * batch_size]
                # empty tensor to hold all of their next states
                next_states = torch.empty(batch_size * 12, 480)
                # populate next_states tensor
                num = 0
                # for each training example
                for cube in training_batch:
                    # print(str(num) + 'th cube', cube.get_array())
                    # make all 12 possible moves and store tensor
                    for idx in range(12):
                        cube.idx_turn(idx)
                        # print('\t', cube.get_array())
                        next_states[num] = cube.to_tensor()
                        cube.idx_turn(idx, True)
                        num += 1
                # get outputs
                outputs = self.model_label(next_states)
                # find label for each training example
                for x in range(batch_size):
                    min_cost = float('inf')
                    for i in range(x * 12, x * 12 + 12):
                        cost = 1 + outputs[i].item()
                        if cost < min_cost:
                            min_cost = cost
                    # store results
                    X[b][x] = examples[b * batch_size + x].to_tensor()
                    Y[b][x] = min_cost
            '''
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
            '''
        return X, Y

    def fit(self, batches):
        pass

    def train(self, num_scrambles, batch_size, max_updates):
        # ensure the number of scrambles can be split into batches evenly
        if(num_scrambles % batch_size != 0):
            print(f'number of scrambles ({num_scrambles}) and batch size ({batch_size}) aren\'t compatible')
            exit()

        num_updates = 0
        for i in range(10): # this should be a while(num_updates < max_updates) once things are working
            samples = generate_training_data_avi(num_scrambles, 30)


            Y = self.label_samples(samples)
            X = self.cubes_to_input(samples)

            batches = self.get_batches(samples, batch_size)

            self.fit(batches)
            if(len(self.seen_states) > STATES_PER_UPDATE):
                self.model_label.load_state_dict(model_train.state_dict())
                self.optim_label.load_state_dict(optim_train.state_dict())
                num_updates += 1
