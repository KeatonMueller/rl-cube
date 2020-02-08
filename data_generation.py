import torch
import random

import cube as C

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
