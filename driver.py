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

            best_val, best_i = (float('-inf'), -1)
            for i in range(len(v_x)):
                if(v_x[i] > best_val):
                    best_val = v_x[i]
                    best_i = i

            # this happens when the weights explode
            if best_i == -1:
                import pdb; pdb.set_trace()

            # get labels for inputs
            y_v = torch.tensor([[best_val]])
            y_p = torch.tensor([best_i])

            X.append(cube.to_tensor())
            Y.append((y_v, y_p))

    return (X, Y)


if __name__ == "__main__":
    PERIODS = 5     # number of times to generate new training data
    EPOCHS = 5      # number of times to evaluate all of the training data per period

    NUM_SCRAMBLES = 50
    SCRAMBLE_LENGTH = 1

    net = CubeNet()
    optimizer = optim.SGD(net.parameters(), lr=0.05)

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
                cube = C.Cube()
                cube.turn(face1, dir1)
                # cube.turn(face2, dir2)

                out_v, out_p = net(cube.to_tensor())
                (f1, d1) = idx_to_move[torch.argmax(out_p).item()]
                cube.turn(f1, d1)

                # out_v, out_p = net(cube.to_tensor())
                # (f2, d2) = idx_to_move[torch.argmax(out_p).item()]
                # cube.turn(f2, d2)

                if(cube.is_solved()):
                    print('SOLVED!')
                else:
                    print(cube)
