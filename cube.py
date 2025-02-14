import torch
import copy
from enum import Enum

import utils.transformations as transformations

'''
    Face enum
    the six outer layers on a Rubik's Cube
'''
class Face(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    FRONT = 4
    BACK = 5

'''
    Dir enum
    either CW for clockwise or CCW for counterclockwise
'''
class Dir(Enum):
    CW = 0
    CCW = 1

'''
    arrays containing the Face and Dir enums
'''
Faces = [ Face.RIGHT, Face.LEFT, Face.UP, Face.DOWN, Face.FRONT, Face.BACK ]
Dirs = [ Dir.CW, Dir.CCW ]

'''
    a map from a Face enum to the corresponding transformation maps needed to
    rotate that face in either direction
'''
FACE_TO_TRANS = {
    Face.RIGHT: (transformations.R_CORNER_TRANS, transformations.R_EDGE_TRANS),
    Face.LEFT: (transformations.L_CORNER_TRANS, transformations.L_EDGE_TRANS),
    Face.UP: (transformations.U_CORNER_TRANS, transformations.U_EDGE_TRANS),
    Face.DOWN: (transformations.D_CORNER_TRANS, transformations.D_EDGE_TRANS),
    Face.FRONT: (transformations.F_CORNER_TRANS, transformations.F_EDGE_TRANS),
    Face.BACK: (transformations.B_CORNER_TRANS, transformations.B_EDGE_TRANS)
}

'''
    a map from move index to the corresponding Face and Dir enums
    the order corresponds to the CubeNet's policy branch output
'''
idx_to_move = {
    0: (Face.RIGHT, Dir.CW),
    1: (Face.RIGHT, Dir.CCW),
    2: (Face.LEFT, Dir.CW),
    3: (Face.LEFT, Dir.CCW),
    4: (Face.UP, Dir.CW),
    5: (Face.UP, Dir.CCW),
    6: (Face.DOWN, Dir.CW),
    7: (Face.DOWN, Dir.CCW),
    8: (Face.FRONT, Dir.CW),
    9: (Face.FRONT, Dir.CCW),
    10: (Face.BACK, Dir.CW),
    11: (Face.BACK, Dir.CCW),
}

'''
    20x24 representation of a Rubik's Cube
    x-axis is the letter notation assigned to each sticker, indicating the 24 possible
        locations for an edge or corner sticker
    y-axis are the 20 stickers we need to keep track of to deterministically track the
        state of all 54 stickers

    this is here mainly for reference
'''
solved_cube_matrix = [
   # A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X
   # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WGO
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WBO
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WBR
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WGR
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # YGR
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # YBR
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # YBO
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # YGO
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WO
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WB
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WR
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # WG
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # YR
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # YB
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # YO
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # YG
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # RB
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # RG
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # OG
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # OB
]

'''
    1x20 representation of a Rubik's Cube
    0 <= arr[i] < 24 for all i
    index refers to one of the 20 stickers needed to determine the state of all 54 stickers
    value at that index refers to the position of that sticker

    order of indices is
        Corners A, B, C, D, U, V, W, X
        Edges A, B, C, D, U, V, W, X, J, L, R, T
    which can also be seen in the matrix format above
'''
solved_cube_arr = [ 0, 1, 2, 3, 20, 21, 22, 23, 0, 1, 2, 3, 20, 21, 22, 23, 9, 11, 17, 19 ]

'''
    device for CPU or GPU calculations
'''
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def arr_to_matrix(arr):
    '''
        returns the 20x24 one-hot encoding representation of the given 1x20 representation

        arr: a 1x20 representation of a Rubik's Cube
    '''
    matrix = []
    for loc in arr:
        matrix.append([1 if i == loc else 0 for i in range(24)])
    return matrix

def get_neighbors(cube):
    '''
        return all 12 cube states reachable from given state
        with an outer layer quarter turn

        cube: Cube object from which to get neighbors
    '''
    neighbors = []
    for idx in range(12):
        neighbor = Cube(cube)
        neighbor.idx_turn(idx)
        neighbors.append(neighbor)
    return neighbors

class Cube():
    def __init__(self, cube=None):
        '''
            initializes a Rubik's Cube

            cube:  a Cube object to copy state from
                   if not provided, the cube is initialized
                   in the solved state
        '''
        if cube:
            self.arr = copy.copy(cube.arr)
        else:
            self.arr = copy.copy(solved_cube_arr)

    def turn(self, face, direction, undo=False):
        '''
            performs any single outer layer turn in any direction

            face: one of the Face enums
            direction: one of the Dir enums
            undo: if True, undo the given move. defaults to False
        '''
        # get transformation maps for this face
        corner_trans, edge_trans = FACE_TO_TRANS[face]

        # convert Dir enum to index into transformation maps
        direction = 0 if direction == Dir.CW else 1
        # swap the direction if asked to undo
        if undo:
            direction = 1 - direction

        # rotate corners
        for i in range(8):
            self.arr[i] = corner_trans[direction][self.arr[i]] if self.arr[i] in corner_trans[direction] else self.arr[i]

        # rotate edges
        for i in range(8, 20):
            self.arr[i] = edge_trans[direction][self.arr[i]] if self.arr[i] in edge_trans[direction] else self.arr[i]

    def idx_turn(self, idx, undo=False):
        '''
            performs any single outer layer turn in any direction

            idx: an index referring to a specific turn
            undo: if True, undo the given move. defaults to False
        '''
        face, direction = idx_to_move[idx]
        self.turn(face, direction, undo)

    def to_tensor(self):
        '''
            returns 1x480 (flattened 20x24) torch tensor representation of the Rubik's Cube
        '''
        flattened = []
        for loc in self.arr:
            flattened.extend([1.0 if i == loc else 0.0 for i in range(24)])
        return torch.tensor([flattened])

    def get_array(self):
        '''
            returns 1x20 array representation of the Rubik's Cube
        '''
        return self.arr

    def is_solved(self):
        '''
            returns True if cube is solved
            returns False otherwise
        '''
        return self.arr == solved_cube_arr

    def reward(self):
        '''
            returns 1 if the cube is solved
            returns -1 otherwise
        '''
        return 1 if self.is_solved() else -1

    def reset(self):
        '''
            resets the cube to the solved state
        '''
        self.arr = copy.copy(solved_cube_arr)

    def __hash__(self):
        '''
            generates a unique hash of the cube state which
            corresponds to a tuple of the state array
        '''
        return hash(tuple(self.arr))

    def __eq__(self, other):
        '''
            assesses equality of two Cube objects
        '''
        if(type(other) is Cube):
            return self.arr == other.arr
        return False
