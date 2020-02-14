import torch
from torch.utils import data
import random

import cube as C

class Dataset(data.Dataset):
    '''
        a PyTorch dataset to utilize parallel generation of data
    '''
    def __init__(self, num, length, net):
        '''
            initialize dataset

            num: number of cubes to scramble in the dataset
            length: length of each scramble
            net: a CubeNet network to label the training examples
        '''
        self.num = num
        self.length = length

        self.cube = C.Cube()
        self.net = net

        self.total = self.num * self.length

    def set_net(self, net):
        '''
            update the CubeNet network used to label samples

            net: a CubeNet network
        '''
        self.net = net

    def __len__(self):
        '''
            returns the total number of training examples
        '''
        return self.total

    def __getitem__(self, idx):
        '''
            returns the item at index idx in the dataset
            since items are random, the idx is only relevant for
            how many turns into a random scramble it is
        '''
        num_turns = idx % self.length
        self.net.eval()
        self.cube.reset()
        for i in range(num_turns):
            # perform a random turn
            self.cube.turn(random.choice(C.Faces), random.choice(C.Dirs))

        # label this position
        # list to store values of 12 children
        v_x = []

        # evaluate 12 children
        for face in C.Faces:
            for dir in C.Dirs:
                # perform one turn
                self.cube.turn(face, dir)
                # evaluate resulting position
                with torch.no_grad():
                    v, p = self.net(self.cube.to_tensor())
                # append value and reward for being in this position
                v_x.append(self.cube.reward() + v.item())
                # undo turn
                self.cube.turn(face, dir, True)

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

        return self.cube.to_tensor(), (y_v, y_p)

def generate_training_data(num, length, net):
    '''
        generates an array of scrambled cubes of length (num * length)
        as well as and array of (value, policy) labels for each cube

        num: number of cubes to scramble
        length: length of scramble per cube
        net: a CubeNet used to label the generated examples
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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
                        v, p = net(cube.to_tensor().to(device))
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
            y_v = torch.tensor([[best_val]]).to(device)
            y_p = torch.tensor([best_i]).to(device)

            # store results
            X.append((cube.to_tensor().to(device), j + 1))
            Y.append((y_v, y_p))

    return (X, Y)
