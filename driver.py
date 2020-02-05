import torch
import torch.optim as optim
import torch.nn.functional as F
import random, copy
import cube as C
from cube_net import CubeNet

'''
    num: number of cubes to scramble
    len: length of scramble per cube
    Net: a CubeNet used to label the generated examples

    generates an array of scrambled cubes of length (num * len)
'''
def generate_training_data(num, length, net):
    X = []
    Y = []
    for i in range(num):
        cube = C.get_cube_arr()
        for j in range(length):
            # perform a random turn
            C.turn(random.choice(C.Faces), random.choice(C.Dirs), cube)

            # list to store values of 12 children
            v_x = []

            # evaluate 12 children
            for face in C.Faces:
                for dir in C.Dirs:
                    # perform one turn
                    C.turn(face, dir, cube)
                    # evaluate resulting position
                    v, p = net(C.arr_to_tensor(cube))
                    # append value and reward for being in this position
                    v_x.append(C.R(cube) + v.item())
                    # undo turn
                    C.turn(face, dir, cube, True)

            best_val, best_i = (float('-inf'), -1)
            for i in range(len(v_x)):
                if(v_x[i] > best_val):
                    best_val = v_x[i]
                    best_i = i

            y_v = torch.tensor([[best_val]])
            # y_p = torch.tensor([[1 if i == best_i else 0 for i in range(len(v_x))]])
            y_p = torch.tensor([best_i])

            X.append(C.arr_to_tensor(cube))
            Y.append((y_v, y_p))

    return (X, Y)


if __name__ == "__main__":
    PERIODS = 10     # number of times to generate new training data
    EPOCHS = 5      # number of times to evaluate all of the training data per period

    net = CubeNet()
    optimizer = optim.Adam(net.parameters())

    for period in range(PERIODS):
        print('period', period)
        print('\tgenerating training data...')
        X, Y = generate_training_data(50, 1, net)
        print('\tdone')

        # print(net.out_v.weight)
        for epoch in range(EPOCHS):
            print('\t\tepoch', epoch, end=' ')
            for x, y in zip(X, Y):
                y_v, y_p = y
                net.zero_grad()
                out_v, out_p = net(x)
                # if epoch == 4:
                #     import pdb; pdb.set_trace()
                loss_v = F.mse_loss(out_v, y_v)
                loss_p = F.cross_entropy(out_p, y_p)

                loss = loss_v + loss_p
                loss.backward()
                optimizer.step()
            print('loss', loss)
            # print(net.out_v.weight)
