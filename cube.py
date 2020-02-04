import turns
import copy
from enum import Enum

class Face(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    FRONT = 4
    BACK = 5

FACE_TO_TRANS = {
    Face.RIGHT: (turns.R_CORNER_TRANS, turns.R_EDGE_TRANS),
    Face.LEFT: (turns.L_CORNER_TRANS, turns.L_EDGE_TRANS),
    Face.UP: (turns.U_CORNER_TRANS, turns.U_EDGE_TRANS),
    Face.DOWN: (turns.D_CORNER_TRANS, turns.D_EDGE_TRANS),
    Face.FRONT: (turns.F_CORNER_TRANS, turns.F_EDGE_TRANS),
    Face.BACK: (turns.B_CORNER_TRANS, turns.B_EDGE_TRANS)
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
    face: one of the Face enums
    dir: 0 for clockwise, 1 for counterclockwise
    cube: a 1x20 representation of a Rubik's Cube

    performs any single outer layer turn in any direction
'''
def turn(face, dir, cube):
    # get transformation map for this face
    corner_trans, edge_trans = FACE_TO_TRANS[face]

    # rotate corners
    for i in range(8):
        cube[i] = corner_trans[dir][cube[i]] if cube[i] in corner_trans[dir] else cube[i]

    # rotate edges
    for i in range(8, 20):
        cube[i] = edge_trans[dir][cube[i]] if cube[i] in edge_trans[dir] else cube[i]

if __name__ == "__main__":
    cube = copy.copy(solved_cube_arr)
    print(cube)
    turn(Face.RIGHT, 0, cube)
    print(cube)
    turn(Face.LEFT, 1, cube)
    print(cube)
    turn(Face.FRONT, 0, cube)
    print(cube)
    turn(Face.UP, 0, cube)
    print(cube)
    turn(Face.DOWN, 1, cube)
    print(cube)
    turn(Face.BACK, 1, cube)
    print(cube)
