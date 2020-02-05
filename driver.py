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
                    with torch.no_grad():
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

            if best_i == -1:
                import pdb; pdb.set_trace()

            y_v = torch.tensor([[best_val]])
            # y_p = torch.tensor([[1 if i == best_i else 0 for i in range(len(v_x))]])
            y_p = torch.tensor([best_i])

            X.append(C.arr_to_tensor(cube))
            Y.append((y_v, y_p))

    return (X, Y)


if __name__ == "__main__":
    PERIODS = 5     # number of times to generate new training data
    EPOCHS = 5      # number of times to evaluate all of the training data per period

    NUM_SCRAMBLES = 50
    SCRAMBLE_LENGTH = 1

    net = CubeNet()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for period in range(PERIODS):
        print('period', period)
        print('\tgenerating training data...')
        X, Y = generate_training_data(NUM_SCRAMBLES, SCRAMBLE_LENGTH, net)
        print('\tdone')

        # print(net.out_v.weight)
        for epoch in range(EPOCHS):
            print('\t\tepoch', epoch, end=' ')
            for x, y in zip(X, Y):
                y_v, y_p = y
                net.zero_grad()
                out_v, out_p = net(x)
                # if epoch == 4 and period == 3:
                #     import pdb; pdb.set_trace()
                loss_v = F.mse_loss(out_v, y_v)
                loss_p = F.cross_entropy(out_p, y_p)

                loss = loss_v + loss_p
                loss.backward()
                optimizer.step()
            print('loss', loss)
            # print(net.out_v.weight)
    idx_to_move = {
        0: (C.Face.RIGHT, C.Dir.CW),
        1: (C.Face.RIGHT, C.Dir.CCW),
        2: (C.Face.LEFT, C.Dir.CW),
        3: (C.Face.LEFT, C.Dir.CCW),
        4: (C.Face.UP, C.Dir.CW),
        5: (C.Face.UP, C.Dir.CCW),
        6: (C.Face.DOWN, C.Dir.CW),
        7: (C.Face.DOWN, C.Dir.CCW),
        8: (C.Face.FRONT, C.Dir.CW),
        9: (C.Face.FRONT, C.Dir.CCW),
        10: (C.Face.BACK, C.Dir.CW),
        11: (C.Face.BACK, C.Dir.CCW),
    }
    print('testing')
    with torch.no_grad():
        for face1 in C.Faces:
            for dir1 in C.Dirs:
                # for face2 in C.Faces:
                #     for dir2 in C.Dirs:
                cube = C.get_cube_arr()
                C.turn(face1, dir1, cube)
                # C.turn(face2, dir2, cube)

                out_v, out_p = net(C.arr_to_tensor(cube))
                (f1, d1) = idx_to_move[torch.argmax(out_p).item()]
                C.turn(f1, d1, cube)

                # out_v, out_p = net(C.arr_to_tensor(cube))
                # (f2, d2) = idx_to_move[torch.argmax(out_p).item()]
                # C.turn(f2, d2, cube)

                if(cube == C.solved_cube_arr):
                    print('SOLVED!')
                else:
                    print(cube)
