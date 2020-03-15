import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy
from math import ceil

import cube as C
from cube_net import ResCubeNet
from data_generation import generate_training_data_avi
from a_star_test import a_star_test, attempt_solve
import progress_printer as prog_print

# see 10,000,000 unique states before updating parameters
STATES_PER_UPDATE = 50000

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class AVI:
    '''
        class to implement approximate value iteration
    '''
    def __init__(self):
        self.model_train = ResCubeNet().to(device)
        self.optim_train = optim.Adam(self.model_train.parameters(), lr=0.01)

        self.model_label = ResCubeNet().to(device)
        self.model_label.load_state_dict(self.model_train.state_dict())

        self.seen_states = set()

    def load(self, PATH):
        '''
            load model checkpoint

            PATH: path to model checkpoint
        '''
        checkpoint = torch.load(PATH, map_location=device)
        self.model_label.load_state_dict(checkpoint['model_state_dict'])
        self.model_train.load_state_dict(checkpoint['model_state_dict'])
        self.optim_train.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_label.eval()
        self.model_train.train()
        print('loaded', PATH)

    def cubes_to_input(self, samples):
        '''
            converts a list of Cube objects into a pytorch tensor for input into a network

            samples: list of Cube objects that are training examples
        '''
        num_samples = len(samples)
        # empty tensor to store input
        X = torch.empty(num_samples, 480)
        # for each cube
        for i, cube in enumerate(samples):
            prog_print.print_progress('converting', i, num_samples)
            # mark this state as seen
            self.seen_states.add(cube)
            # convert it to a tensor
            X[i] = cube.to_tensor()
        prog_print.print_progress_done('converted', num_samples)
        return X

    def label_samples(self, samples, label_batch_size):
        '''
            label the given training examples

            samples: list of Cube objects that are training examples
            label_batch_size: number of samples to put through the network at a time
        '''
        # empty tensor to store labels
        Y = torch.empty(len(samples), 1)
        # number of batches
        num_batches = ceil(len(samples) / label_batch_size)
        # empty tensor to store next states - input to network
        next_states = torch.empty(label_batch_size * 12, 480, device=device)
        # calculate number of samples
        num_samples = len(samples)

        self.model_label.eval()
        with torch.no_grad():
            for b in range(num_batches):
                # print progress
                prog_print.print_progress('labelling', min(num_samples, b * label_batch_size + label_batch_size), num_samples)
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
            # print completed progress
            prog_print.print_progress_done('labelled', num_samples)
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

        # reshape input and output into batches
        X = X.view(-1, batch_size, 480)
        Y = Y.view(-1, batch_size, 1)

        return X, Y

    def fit(self, epochs, batches):
        '''
            fit model_train on the given data

            epochs: number of epochs to train for
            batches: training data
        '''
        # decouple the input and output batches
        X_batches, Y_batches = batches
        # determine batch size
        batch_size = X_batches.size()[1]
        # compute total number of scrambles (for progress printing)
        total = X_batches.size()[0] * X_batches.size()[1]
        # train model
        self.model_train.train()
        for e in range(epochs):
            # keep track of number of batches trained (for printing purposes)
            num = 0
            # for each batch
            for x_batch, y_batch in zip(X_batches, Y_batches):
                # print progress
                prog_print.print_progress(('epoch: ' + str(e)), num*batch_size+batch_size, total)
                num += 1
                # send batch to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # zero out the gradient
                self.model_train.zero_grad()
                # get the outputs
                output = self.model_train(x_batch)
                # compute loss
                loss = F.mse_loss(y_batch, output)
                # backprop
                loss.backward()
                # update weights
                self.optim_train.step()
            # print final loss after finishing epoch
            prog_print.print_progress_done(('epoch: ' + str(e)), total, end=('loss: ' + str(loss.item())))

    def train(self, epochs, num_scrambles, batch_size, label_batch_size, max_updates):
        # check that the number of scrambles can be split into batches evenly
        # (this is up to the caller to ensure)
        if(num_scrambles % batch_size != 0):
            print(f'number of scrambles ({num_scrambles}) and batch size ({batch_size}) aren\'t compatible')
            exit()

        num_updates = 0
        # until we've hit the desired number of updates
        while(num_updates < max_updates):
            # generate Cube object samples
            samples = generate_training_data_avi(num_scrambles, 30)

            # convert them to tensors for input to network
            X = self.cubes_to_input(samples)
            # generate labels for the samples
            Y = self.label_samples(samples, label_batch_size)
            # convert data into batches
            batches = self.get_batches(X, Y, batch_size)
            # train model
            self.fit(epochs, batches)
            # report number of unique cube states encountered so far
            print('seen', len(self.seen_states), 'states')
            # if number exceeds threshold, update model_label
            if(len(self.seen_states) > STATES_PER_UPDATE):
                print('updating model_label')
                self.model_label.load_state_dict(self.model_train.state_dict())
                self.seen_states = set()
                num_updates += 1

    def test(self, tests, scramble_length, time_limit):
        '''
            perform requested tests of the model

            tests: list of test names
            scramble_length: length of scrambles to test
            time_limit: time to attempt each solve
        '''
        if('a_star' in tests):
            a_star_test(self.model_label, scramble_length, time_limit)

    def solve(self, cube, time_limit, scramble):
        attempt_solve(self.model_label, cube, time_limit, None, scramble)

    def save(self, PATH):
        '''
            save model checkpoint

            PATH: location to save model checkpoint
        '''
        torch.save({
            'model_state_dict': self.model_train.state_dict(),
            'optimizer_state_dict': self.optim_train.state_dict()
        }, PATH)
        print('saved model as', PATH)
