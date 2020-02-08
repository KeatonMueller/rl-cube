import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy
import argparse

import cube as C
from cube_net import CubeNet
from naive_test import naive_test
from mcts_test import mcts_test

def generate_training_data(num, length, net):
    '''
        generates an array of scrambled cubes of length (num * length)
        as well as and array of (value, policy) labels for each cube

        num: number of cubes to scramble
        length: length of scramble per cube
        net: a CubeNet used to label the generated examples
    '''
    # enter eval mode (network shouldn't be training during this)
    net.eval()
    # lists for generated input and labels
    X = []
    Y = []
    cube = C.Cube()
    # for `num` number of cubes
    for i in range(num):
        cube.reset()
        # for `length` number of turns
        for j in range(length):
            # perform a random turn
            cube.turn(random.choice(C.Faces), random.choice(C.Dirs))

            # list to store values of 12 children
            v_x = []

            # evaluate 12 children
            for face in C.Faces:
                for dir in C.Dirs:
                    # perform one turn
                    cube.turn(face, dir)
                    # evaluate resulting position
                    with torch.no_grad():
                        v, p = net(cube.to_tensor())
                    # append value and reward for being in this position
                    v_x.append(cube.reward() + v.item())
                    # undo turn
                    cube.turn(face, dir, True)

            # perform max and argmax
            best_val, best_i = (float('-inf'), -1)
            for i in range(len(v_x)):
                if(v_x[i] > best_val):
                    best_val = v_x[i]
                    best_i = i

            # this happens when the weights become nan
            if best_i == -1:
                import pdb; pdb.set_trace()

            # get labels for inputs
            y_v = torch.tensor([[best_val]])
            y_p = torch.tensor([best_i])

            # store results
            X.append(cube.to_tensor())
            Y.append((y_v, y_p))

    return (X, Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run rl-cube')
    parser.add_argument('-p', '--periods', type=int, default=5, help='number of times to generate new training data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of times to evaluate all of the training data per period')
    parser.add_argument('-n', '--number', type=int, default=50, help='number of cubes to scramble per data generation')
    parser.add_argument('-l', '--length', type=int, default=2, help='length of each scramble')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.01, help='learning rate of optimizer')
    parser.add_argument('-t', '--test', type=str, nargs='+', default='[naive]', help='what kinds of tests to run (like `naive` or `mcts`)')
    parser.add_argument('--load', metavar='PATH', type=str, help='load model parameters from a file')
    parser.add_argument('--save', metavar='PATH', type=str, help='save model parameters to a file')
    args = parser.parse_args()

    # number of times to generate new training data
    PERIODS = args.periods
    # number of times to evaluate all of the training data per period
    EPOCHS = args.epochs
    # number of scrambles per data generation
    NUM_SCRAMBLES = args.number
    # length of each scramble
    SCRAMBLE_LENGTH = args.length
    # learning rate of optimizer
    LR = args.learning_rate

    # initialize CubeNet and optimizer
    net = CubeNet()
    optimizer = optim.SGD(net.parameters(), lr=LR)

    # load model parameters if arg exists
    if(args.load):
        # load model checkpoint
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.eval()

    # otherwise train model
    else:
        # loop over the periods
        for period in range(PERIODS):
            print('period:', period)
            # generate new training data
            X, Y = generate_training_data(NUM_SCRAMBLES, SCRAMBLE_LENGTH, net)

            # enter training mode
            net.train()
            # loop over the epochs
            for epoch in range(EPOCHS):
                print('\t\tepoch:', epoch, end='\t')
                for x, y in zip(X, Y):
                    # get expected output
                    y_v, y_p = y

                    # calculate network's output
                    net.zero_grad()
                    out_v, out_p = net(x)

                    # compute and sum loss
                    loss_v = F.mse_loss(out_v, y_v)
                    loss_p = F.cross_entropy(out_p, y_p)
                    loss = loss_v + loss_p

                    # update weights
                    loss.backward()
                    optimizer.step()

                # report loss
                print('loss:', loss.item())

    if('none' not in args.test):
        if('naive' in args.test):
            # run a naive test
            naive_test(net, SCRAMBLE_LENGTH)
        if('mcts' in args.test):
            # run a mcts test
            mcts_test(net, SCRAMBLE_LENGTH, 5)
    # save model if arg exists
    if(args.save):
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.save)
