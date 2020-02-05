import torch
import transformations
import copy
from enum import Enum

class Face(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    FRONT = 4
    BACK = 5

class Dir(Enum):
    CW = 0
    CCW = 1

Faces = [ Face.RIGHT, Face.LEFT, Face.UP, Face.DOWN, Face.FRONT, Face.BACK ]
Dirs = [ Dir.CW, Dir. CCW ]

FACE_TO_TRANS = {
    Face.RIGHT: (transformations.R_CORNER_TRANS, transformations.R_EDGE_TRANS),
    Face.LEFT: (transformations.L_CORNER_TRANS, transformations.L_EDGE_TRANS),
    Face.UP: (transformations.U_CORNER_TRANS, transformations.U_EDGE_TRANS),
    Face.DOWN: (transformations.D_CORNER_TRANS, transformations.D_EDGE_TRANS),
    Face.FRONT: (transformations.F_CORNER_TRANS, transformations.F_EDGE_TRANS),
    Face.BACK: (transformations.B_CORNER_TRANS, transformations.B_EDGE_TRANS)
}
'''
    20x24 representation of a Rubik's Cube
    x-axis is the letter notation assigned to each sticker, indicating the 24 possible
        locations for an edge or corner sticker
    y-axis are the 20 stickers we need to keep track of to deterministically track the
        state of all 54 stickers
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
    arr: a 1x20 representation of a Rubik's Cube

    returns the 20x24 one-hot encoding representation of the given 1x20 representation
'''
def arr_to_matrix(arr):
    matrix = []
    for loc in arr:
        matrix.append([1 if i == loc else 0 for i in range(24)])
    return matrix


'''
    arr: a 1x20 representation of a Rubik's Cube

    returns a 1x480 (flattened 20x24) torch tensor representation of a Rubik's Cube
'''
def arr_to_tensor(arr):
    tensor = torch.tensor([[]])
    for loc in arr:
        line = torch.tensor([[1.0 if i == loc else 0.0 for i in range(24)]])
        tensor = torch.cat((tensor, line), 1)
    return tensor

'''
    face: one of the Face enums
    dir: one of the Dir enums
    cube: a 1x20 representation of a Rubik's Cube
    undo: a boolean. True to undo the requested turn, False to perform it normally

    performs any single outer layer turn in any direction
'''
def turn(face, dir, cube, undo=False):
    # get transformation map for this face
    corner_trans, edge_trans = FACE_TO_TRANS[face]

    # convert Dir enum to index into transformation maps
    dir = 0 if dir == Dir.CW else 1
    # swap the direction if asked to undo
    if undo:
        dir = 1 - dir

    # rotate corners
    for i in range(8):
        cube[i] = corner_trans[dir][cube[i]] if cube[i] in corner_trans[dir] else cube[i]

    # rotate edges
    for i in range(8, 20):
        cube[i] = edge_trans[dir][cube[i]] if cube[i] in edge_trans[dir] else cube[i]

'''
    returns a shallow copy of solved_cube_arr
'''
def get_cube_arr():
    return copy.copy(solved_cube_arr)


'''
    cube: a 1x20 representation of a Rubik's Cube

    returns 1 if the given cube is solved, -1 otherwise
'''
def R(cube):
    return 1 if cube == solved_cube_arr else -1


if __name__ == "__main__":
    cube = get_cube_arr()
    print(cube)
    turn(Face.RIGHT, Dir.CCW, cube)
    print(cube)
    turn(Face.LEFT, Dir.CW, cube)
    print(cube)
    turn(Face.FRONT, Dir.CCW, cube)
    print(cube)
    turn(Face.UP, Dir.CCW, cube)
    print(cube)
    turn(Face.DOWN, Dir.CW, cube)
    print(cube)
    turn(Face.BACK, Dir.CW, cube)
    print(cube)
