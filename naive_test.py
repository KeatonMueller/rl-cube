import torch

import cube as C

'''
    a map from move index to the corresponding Face and Dir enums
    the order corresponds to the CubeNet's policy branch output
'''
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

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def naive_test(net, length):
    '''
        naively tests (that is, uses the policy network alone without MCTS)
        the network on every scramble of the requested length

        net: a CubeNet network
        length: the length of the scramble to attempt to solve
    '''
    stats = {
        'hits': 0,
        'total': 0
    }
    naive_test_helper(net, C.Cube(), length, length, stats)
    print('naive test: solved', stats['hits'], 'out of', stats['total'], '(' + str(round(stats['hits'] / stats['total'] * 100, 2)) + '%)', str(length) + '-move scrambles')

def naive_test_helper(net, cube, curr_len, orig_len, stats):
    '''
        performs all 12 possible moves and either attempts a solve
        afterwards or recursively calls itself to further scramble the cube

        net: a CubeNet network
        cube: a Cube object currently being scrambled
        curr_len: the number of turns remaining in the scramble
        orig_len: the number of turns in the eventual scramble
        stats: a dict tracking solved cubes and total attempts
    '''
    for face_ in C.Faces:
        for dir_ in C.Dirs:
            # make a turn
            cube.turn(face_, dir_)
            if(curr_len == 1):
                # if no more turns needed, attempt a solve
                attempt_solve(net, C.Cube(cube), orig_len, stats)
            else:
                # otherwise recurse and keep scrambling
                naive_test_helper(net, cube, curr_len-1, orig_len, stats)
            # undo the turn
            cube.turn(face_, dir_, True)

def attempt_solve(net, cube, length, stats):
    '''
        attempts to solve the given cube using the given network using
        the given number of turns

        net: a CubeNet network used to attempt the solve
        cube: a Cube object that we will try to solve
        length: the number of turns used to scramble the cube
        stats: a dict tracking solved cubes and total attempts
    '''
    with torch.no_grad():
        for i in range(length):
            # run the cube through the network and do the turn the policy says to
            out_v, out_p = net(cube.to_tensor().to(device))
            f, d = idx_to_move[torch.argmax(out_p).item()]
            cube.turn(f, d)

    # check if solved; update stats
    if(cube.is_solved()):
        stats['hits'] += 1
    stats['total'] += 1
