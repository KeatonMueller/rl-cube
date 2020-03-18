import torch
from torch.utils import data
import random

import cube as C
import progress_printer as prog_print

'''
    device for CPU or GPU calculations
'''
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def generate_training_data_adi(num, length, net):
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
    total_num = num * length
    print_len = len(str(total_num)) * 2 + 1
    with torch.no_grad():
        # for `num` number of cubes
        for i in range(num):
            cube.reset()
            # for `length` number of turns
            for j in range(length):
                print(('generating {0:>'+str(print_len)+'}').format(str(i*length+j)+'/'+str(total_num)), end='\r')
                # perform a random turn
                cube.turn(random.choice(C.Faces), random.choice(C.Dirs))

                # list to store values of 12 children
                v_x = []

                # evaluate 12 children
                best_val, best_move = (float('-inf'), -1)
                for move in range(12):
                    # perform one turn
                    cube.idx_turn(move)
                    # evaluate resulting position
                    v, p = net(cube.to_tensor().to(device))
                    value = v.item() + cube.reward()
                    # update best
                    if(value > best_val):
                        best_val = value
                        best_move = move
                    # undo turn
                    cube.idx_turn(move, True)

                # this happens when the weights become nan
                if best_move == -1:
                    import pdb; pdb.set_trace()

                # create labels for inputs
                y_v = torch.tensor([[best_val]])
                y_p = torch.tensor([best_move])

                # store results
                X.append((cube.to_tensor(), j + 1))
                Y.append((y_v, y_p))
    print('generated ' + str(total_num) + '/' + str(total_num))
    return (X, Y)


def generate_training_data_avi(num_scrambles, max_scramble_len):
    '''
        generates training data of scrambled cubes for value iteration
        returns a num_scrambles length array of Cube objects

        num_scrambles: number of cubes to scramble
        max_scramble_len: max length of turns per scramble
    '''
    # tensor for generated input and labels
    X = []
    cube = C.Cube()

    # for `num_scrambles` number of cubes
    for i in range(num_scrambles):
        # print progress
        prog_print.print_progress('\tgenerating', i, num_scrambles)
        cube.reset()
        # make a random number of turns between 1 and max_scramble_len
        scramble_len = random.randint(1, max_scramble_len)
        for j in range(scramble_len):
            # perform a random turn
            cube.idx_turn(random.randint(0, 11))
        # store cube state
        X.append(C.Cube(cube))
    # print completed progress
    prog_print.print_progress_done('\tgenerated', num_scrambles)
    return X
