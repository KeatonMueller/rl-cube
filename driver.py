import torch
import random, copy
import cube as C
from cube_net import CubeNet

'''
    num: number of cubes to scramble
    len: length of scramble per cube
    Net: a CubeNet used to label the generated examples

    generates an array of scrambled cubes of length (num * len)
'''
def generate_training_data(num, length, Net):
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
                    v, p = Net.forward(C.arr_to_tensor(cube))
                    # append value and reward for being in this position
                    v_x.append(C.R(cube) + v.item())
                    # undo turn
                    C.turn(face, dir, cube, True)

            best_val, best_i = (float('-inf'), -1)
            for i in range(len(v_x)):
                if(v_x[i] > best_val):
                    best_val = v_x[i]
                    best_i = i

            y_v = best_val
            y_p = [1 if i == best_i else 0 for i in range(len(v_x))]

            print(y_v)
            print(y_p)

            x = C.arr_to_tensor(cube)
            X.append(x)
            out_v, out_p = Net.forward(x)
            # print('out_v', out_v)
            # print('out_p', out_p)
    return (X, Y)


if __name__ == "__main__":
    Net = CubeNet()
    X, Y = generate_training_data(1, 1, Net)
    # print(X)
