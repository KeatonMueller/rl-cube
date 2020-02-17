import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import random, copy
import argparse

import cube as C
from cube_net import CubeNet
from data_generation import generate_training_data, Dataset
from naive_test import naive_test
from mcts_test import mcts_test, attempt_solve

move_to_idx = {
    'R'  : 0,
    'R\'': 1,
    'L'  : 2,
    'L\'': 3,
    'U'  : 4,
    'U\'': 5,
    'D'  : 6,
    'D\'': 7,
    'F'  : 8,
    'F\'': 9,
    'B'  : 10,
    'B\'': 11,
}

def get_scramble():
    '''
        prompts user for a scramble
        replaces all half turns with two quarter turns
        validates scramble
    '''
    while(True):
        scramble = input('Enter scramble: ').strip().upper()
        if(scramble == ''):
            return None
        # replace double moves with two single moves
        for move in ['R', 'L', 'U', 'D', 'F', 'B']:
            scramble = scramble.replace(move+'2', move + ' ' + move)
        scramble = scramble.split(' ')
        # validate scramble
        for move in scramble:
            if move not in move_to_idx:
                print('Invalid scramble')
                break
        else:
            return scramble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run rl-cube')
    parser.add_argument('-p', '--periods', type=int, default=5, help='number of times to generate new training data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of times to evaluate all of the training data per period')
    parser.add_argument('-n', '--number', type=int, default=50, help='number of cubes to scramble per data generation')
    parser.add_argument('-l', '--length', type=int, default=2, help='length of each scramble')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size during training')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.01, help='learning rate of optimizer')
    parser.add_argument('-t', '--test', type=str, nargs='+', default='[none]', help='what kinds of tests to run (like `naive` or `mcts`)')
    parser.add_argument('--load', metavar='PATH', type=str, help='load model parameters from a file')
    parser.add_argument('--save', metavar='PATH', type=str, help='save model parameters to a file')
    parser.add_argument('--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--solve', action='store_true', default=False, help='run interactive solve mode')
    args = parser.parse_args()

    # number of times to generate new training data
    PERIODS = args.periods
    # number of times to evaluate all of the training data per period
    EPOCHS = args.epochs
    # number of scrambles per data generation
    NUM_SCRAMBLES = args.number
    # length of each scramble
    SCRAMBLE_LENGTH = args.length
    # batch size
    BATCH_SIZE = args.batch
    # learning rate of optimizer
    LR = args.learning_rate

    # device for CPU or GPU calculations
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # initialize CubeNet and optimizer
    net = CubeNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=LR)

    # load model
    if(args.load):
        # load model checkpoint
        checkpoint = torch.load(args.load, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.eval()

    # train model
    if(args.train):
        # loop over the periods
        for period in range(PERIODS):
            print('period:', period)
            # generate new training data
            X, Y = generate_training_data(NUM_SCRAMBLES, SCRAMBLE_LENGTH, net)
            '''
            dataset = Dataset(NUM_SCRAMBLES, SCRAMBLE_LENGTH, net)
            data_generator = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
            '''
            # enter training mode
            net.train()
            # loop over the epochs
            for epoch in range(EPOCHS):
                print('\t\tepoch:', epoch, end='\t')
                for x, y in zip(X, Y):
                # for x, y in data_generator:
                    # get expected output
                    x, distance = x
                    y_v, y_p = y
                    '''
                    # import pdb; pdb.set_trace()
                    y_v, y_p = zip(y)
                    y_v = y_v[0]
                    y_p = y_p[0]
                    y_p = torch.squeeze(y_p, dim=1)
                    '''
                    # calculate network's output
                    net.zero_grad()
                    out_v, out_p = net(x)
                    # out_p = torch.squeeze(out_p, dim=1)

                    # compute and sum loss
                    loss_v = F.mse_loss(out_v, y_v)
                    loss_p = F.cross_entropy(out_p, y_p)
                    loss = loss_v + loss_p
                    loss = loss * (1 / distance)
                    # update weights
                    loss.backward()
                    optimizer.step()

                # report loss
                print('loss:', loss.item())

    # test model
    if('none' not in args.test):
        if('naive' in args.test):
            # run a naive test
            naive_test(net, SCRAMBLE_LENGTH)
        if('mcts' in args.test):
            # run a mcts test
            mcts_test(net, SCRAMBLE_LENGTH, 5)

    # interactive solve mode
    if(args.solve):
        cube = C.Cube()
        scramble = get_scramble()
        while(scramble):
            cube.reset()
            for move in scramble:
                cube.idx_turn(move_to_idx[move])
            attempt_solve(net, cube, 30, None, ' '.join(scramble))
            scramble = get_scramble()

    # save model
    if(args.save):
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.save)
