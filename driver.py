import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy
import cube as C
from cube_net import CubeNet
import naive_test

def generate_training_data(num, length, net):
    '''
        generates an array of scrambled cubes of length (num * length)

        num: number of cubes to scramble
        length: length of scramble per cube
        net: a CubeNet used to label the generated examples
    '''
    X = []
    Y = []
    cube = C.Cube()
    for i in range(num):
        cube.reset()
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
    PERIODS = 5     # number of times to generate new training data
    EPOCHS = 10      # number of times to evaluate all of the training data per period

    NUM_SCRAMBLES = 50      # number of scrambles per data generation
    SCRAMBLE_LENGTH = 2     # length of each scramble

    net = CubeNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for period in range(PERIODS):
        print('period', period)
        # generate new training data
        X, Y = generate_training_data(NUM_SCRAMBLES, SCRAMBLE_LENGTH, net)

        for epoch in range(EPOCHS):
            print('\t\tepoch', epoch, end=' ')
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

            print('loss', loss)

    # run a naive test of the network
    naive_test.naive_test(net, SCRAMBLE_LENGTH)
