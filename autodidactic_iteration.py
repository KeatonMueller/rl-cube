import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import random

import cube as C
from cube_net import CubeNet
from data_generation import generate_training_data_adi
from naive_test import naive_test
from mcts_test import mcts_test, attempt_solve
import progress_printer as prog_print

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class ADI:
    '''
        class to implement autodidactic iteration
    '''
    def __init__(self, LR):
        '''
            initialize model and optimizer

            LR: learning rate of optimizer
        '''
        self.model = CubeNet().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR)

    def load(self, PATH):
        '''
            load model checkpoint

            PATH: location of checkpoint
        '''
        checkpoint = torch.load(PATH, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        print('loaded', PATH)

    def fit(self, PERIODS, EPOCHS, NUM_SCRAMBLES, SCRAMBLE_LENGTH):
        '''
            trains the given network with the given parameters
        '''
        # animated progress bar of epoch training progress
        progress_bar_len = 18
        # report device being trained with
        print('training with', device)
        # allow torch to optimize algorithms
        torch.backends.cudnn.benchmark = True
        # loop over the periods
        for period in range(PERIODS):
            print('period:', period)
            # generate new training data
            X, Y = generate_training_data_adi(NUM_SCRAMBLES, SCRAMBLE_LENGTH, self.model)
            train_len = len(X)

            # enter training mode
            self.model.train()
            # loop over the epochs
            for epoch in range(EPOCHS):
                # print('\tepoch:', epoch, end='\t')
                for i, (x, y) in enumerate(zip(X, Y)):
                    # print progress
                    prog_print.print_progress(('\tepoch: ' + str(epoch)), i, train_len)
                    # get expected output
                    x, distance = x
                    y_v, y_p = y
                    # send them to the device
                    x = x.to(device)
                    y_v = y_v.to(device)
                    y_p = y_p.to(device)

                    # calculate network's output
                    self.model.zero_grad()
                    out_v, out_p = self.model(x)
                    # out_p = torch.squeeze(out_p, dim=1)

                    # compute and sum loss
                    loss_v = F.mse_loss(out_v, y_v)
                    loss_p = F.cross_entropy(out_p, y_p)
                    loss = loss_v + loss_p
                    loss = loss * (4 / (distance + 3))
                    # update weights
                    loss.backward()
                    self.optimizer.step()

                # print completed progress and report loss
                prog_print.print_progress_done(('\tepoch: ' + str(epoch)), train_len, end=('loss: ' + str(loss.item())))

    def test(self, tests, SCRAMBLE_LENGTH, TIME_LIMIT):
        '''
            perform requested tests of the model

            tests: list of test names
            SCRAMBLE_LENGTH: length of scrambles for tests
            TIME_LIMIT: time to attempt each solve
        '''
        if('naive' in tests):
            # run a naive test
            naive_test(self.model, SCRAMBLE_LENGTH)
        if('mcts' in tests):
            # run a mcts test
            mcts_test(self.model, SCRAMBLE_LENGTH, TIME_LIMIT)

    def solve(self, cube, TIME_LIMIT, scramble):
        attempt_solve(self.model, cube, TIME_LIMIT, None, scramble)


    def save(self, PATH):
        '''
            save model checkpoint

            PATH: location to save checkpoint
        '''
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, PATH)
        print('saved model as', PATH)
