import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import argparse
import sys

import cube as C
from autodidactic_iteration import ADI
from approximate_value_iteration import AVI

# global variables
args = None
adi = None
avi = None

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

def main():
    global args, adi, avi
    parser = argparse.ArgumentParser(description = 'Run rl-cube')
    parser.add_argument('-p', '--periods', type=int, default=1, help='number of times to generate new training data')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of times to evaluate all of the training data per period')
    parser.add_argument('-n', '--num_cubes', type=int, default=100, help='number of cubes to scramble per data generation (adi)')
    parser.add_argument('-s', '--num_scrambles', type=int, default=50000, help='number of scrambles per data generation (avi)')
    parser.add_argument('-l', '--length', type=int, default=30, help='length of each scramble')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.01, help='learning rate of optimizer')
    parser.add_argument('-t', '--test', type=str, nargs='+', default='[none]', help='what kinds of tests to run (like `naive` or `mcts`)')
    parser.add_argument('-mu', '--max_updates', type=int, default=100, help='maximum number of times to update the labelling model during AVI')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='batch size during training')
    parser.add_argument('-lb', '--label_batch_size', type=int, default=7200, help='batch size for labelling training examples')
    parser.add_argument('-avi', '--approximate_value_iteration', action='store_true', default=False, help='use approximate value iteration method')
    parser.add_argument('--limit', type=int, default=5, help='time limit (in seconds) for each mcts solve attempt')
    parser.add_argument('--load', metavar='PATH', type=str, help='load model parameters from a file')
    parser.add_argument('--save', metavar='PATH', type=str, help='save model parameters to a file')
    parser.add_argument('--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--solve', action='store_true', default=False, help='run interactive solve mode')
    args = parser.parse_args()

    # number of times to generate new training data
    PERIODS = args.periods
    # number of times to evaluate all of the training data per period
    EPOCHS = args.epochs
    # number of cubes per data generation (adi)
    NUM_CUBES = args.num_cubes
    # number of scrambles per data generation (avi)
    NUM_SCRAMBLES = args.num_scrambles
    # length of each scramble
    SCRAMBLE_LENGTH = args.length
    # time limit for mcts solve attempt
    TIME_LIMIT = args.limit
    # batch size
    BATCH_SIZE = args.batch_size
    # label batch size
    LABEL_BATCH_SIZE = args.label_batch_size
    # max updates
    MAX_UPDATES = args.max_updates
    # learning rate of optimizer
    LR = args.learning_rate

    # device for CPU or GPU calculations
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # if using the default autodidactic iteration method
    if(not args.approximate_value_iteration):
        # initialize CubeNet and optimizer
        adi = ADI(LR)

        # load model
        if(args.load):
            adi.load(args.load)

        # train model
        if(args.train):
            adi.fit(PERIODS, EPOCHS, NUM_CUBES, SCRAMBLE_LENGTH)

        # test model
        if('none' not in args.test):
            adi.test(args.test, SCRAMBLE_LENGTH, TIME_LIMIT)

        # interactive solve mode
        if(args.solve):
            cube = C.Cube()
            scramble = get_scramble()
            while(scramble):
                cube.reset()
                for move in scramble:
                    cube.idx_turn(move_to_idx[move])
                adi.solve(cube, TIME_LIMIT, ' '.join(scramble))
                scramble = get_scramble()

        # save model
        if(args.save):
            adi.save(args.save)

    # if using the value iteration method
    else:
        avi = AVI()
        # load model
        if(args.load):
            avi.load(args.load)

        # train model
        if(args.train):
            avi.train(EPOCHS, NUM_SCRAMBLES, BATCH_SIZE, LABEL_BATCH_SIZE, MAX_UPDATES)

        # test model
        if('none' not in args.test):
            avi.test(args.test, SCRAMBLE_LENGTH, TIME_LIMIT)

        # interactive solve mode
        if(args.solve):
            cube = C.Cube()
            scramble = get_scramble()
            while(scramble):
                cube.reset()
                for move in scramble:
                    cube.idx_turn(move_to_idx[move])
                avi.solve(cube, TIME_LIMIT, ' '.join(scramble))
                scramble = get_scramble()

        # save model
        if(args.save):
            avi.save(args.save)

def solve_scramble(scramble):
    cube = C.Cube()
    scramble = scramble.split(' ')
    for move in scramble:
        cube.idx_turn(move_to_idx[move])
    avi.solve(cube, TIME_LIMIT, ' '.join(scramble))

def avi_init():
    global avi
    avi = AVI()
    avi.load('models\\model_avi_82.lckpt')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\ninterrupted')
        if(not args.approximate_value_iteration):
            # save the interrupted model if it was supposed to be saved
            if(args.save):
                adi.save('model_interrupt_adi')
        else:
            # save the interrupted model if it was supposed to be saved
            if(args.save):
                avi.save('', True)
        sys.exit(0)
else:
    # if not being run as main, initialize AVI
    avi_init()
