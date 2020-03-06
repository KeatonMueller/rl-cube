import torch
import random, math
from math import sqrt, log
from time import time

import cube as C

# exploration hyperparameter
HYP_C = 1.4
# virtual loss hyperparameter
HYP_V = 1000

# exploration hyperparameter for UCT
HYP_UCT = 6

'''
    a map from move index to the string representation
    of the move, for debugging purposes
'''
idx_to_str = {
    0: 'R',
    1: 'R\'',
    2: 'L',
    3: 'L\'',
    4: 'U',
    5: 'U\'',
    6: 'D',
    7: 'D\'',
    8: 'F',
    9: 'F\'',
    10: 'B',
    11: 'B\'',
}

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Tree:
    def __init__(self, cube, net):
        '''
            initializes the tree data structure for MCTS

            cube: a Cube object to be stored at the root
            net: a CubeNet used to guide tree traversal
        '''
        self.map = dict()
        self.net = net
        self.root = Node(cube, self.map, self.net)
        self.map[cube] = self.root

class Node:
    def __init__(self, cube, map, net):
        '''
            initializes a node in the tree

            cube: a Cube object to be stored at this node
            map: a map from Cube objects to their corresponding node
            net: a CubeNet used to guide tree traversal
        '''
        self.children = [ None for i in range(12) ]
        self.child_pos_to_stats = dict()

        self.parents = []

        self.chosen_parent = None
        self.chosen_action = None

        self.T = 0

        # number of times action a has been taken from this state
        self.N = [ 0 for a in range(12) ]
        # maximal value of action a from this state
        self.W = [ 0 for a in range(12) ]
        # virtual loss for action a from this state
        self.L = [ 0 for a in range(12) ]

        net.eval()
        with torch.no_grad():
            out_v, out_p = net(cube.to_tensor().to(device))
        # prior probability of action a from this state
        self.P = out_p[0]

        self.cube = cube
        self.map = map
        self.net = net

        self.is_terminal = cube.is_solved()
        self.is_fully_expanded = self.is_terminal

def mcts_test(net, length, time_limit):
    '''
        uses MCTS to test the network on every scramble of the requested length

        net: a CubeNet network
        length: the length of the scramble to attempt to solve
        time_limit: the time limit for each mcts attempt at solving a cube
    '''
    stats = {
        'hits': 0,
        'total': 0,
        'time': 0,
        'max': -1,
        'fails': []
    }
    # only run exhaustive test if scramble length < 5
    if(length < 5):
        mcts_test_helper_all(net, C.Cube(), length, length, stats, time_limit, '')
    # otherwise randomly scramble the cube `length` times
    else:
        mcts_test_helper_random(net, C.Cube(), length, stats, time_limit)

    # print out results of test
    print('mcts test: solved', stats['hits'], 'out of', stats['total'], \
            '(' + str(round(stats['hits'] / stats['total'] * 100, 2)) + '%)',\
            str(length) + '-move scrambles')
    print('\tmax time: ' + str(round(stats['max'], 2)) + '\n' + \
          '\tavg time: ' + str(round(stats['time'] / stats['hits'], 2)))

    # record failed solves
    if(len(stats['fails']) > 0):
        with open('failed_solves.txt', 'w') as out_file:
            out_file.write(str(stats['fails']))

def mcts_test_helper_all(net, cube, curr_len, orig_len, stats, time_limit, curr_scramble):
    '''
        performs all 12 possible moves and either attempts a solve
        afterwards or recursively calls itself to further scramble the cube

        net: a CubeNet network
        cube: a Cube object currently being scrambled
        curr_len: the number of turns remaining in the scramble
        orig_len: the number of turns in the eventual scramble
        stats: a dict tracking solved cubes and total attempts
        time_limit: the time limit for each mcts attempt at solving a cube
    '''
    for idx in range(12):
        # make a turn
        cube.idx_turn(idx)
        if(curr_len == 1):
            # if no more turns needed, attempt a solve
            attempt_solve(net, C.Cube(cube), time_limit, stats, curr_scramble + ' ' + idx_to_str[idx])
        else:
            # otherwise recurse and keep scrambling
            mcts_test_helper_all(net, cube, curr_len-1, orig_len, stats, time_limit, curr_scramble + ' ' + idx_to_str[idx])
        # undo the turn
        cube.idx_turn(idx, True)

def mcts_test_helper_random(net, cube, length, stats, time_limit):
    '''
        randomly scrambles 1000 cubes `length` times and then attempts
        to solve them

        net: a CubeNet network
        cube: a Cube object currently being scrambled
        length: the number of turns for each scramble
        stats: a dict tracking solved cubes and total attempts
        time_limit: the time limit for each mcts attempt at solving a cube
    '''
    idx_to_inv = { 0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10 }
    # try to solve 1000 randomly scrambled cubes
    for i in range(1000):
        cube.reset()
        curr_scramble = ''
        # make `length` random turns, preventing moves that undo the previous
        prev_idx = -1
        for j in range(length):
            idx = random.randint(0, 11)
            while(idx_to_inv[idx] == prev_idx):
                idx = random.randint(0, 11)
            cube.idx_turn(idx)
            prev_idx = idx
            curr_scramble = curr_scramble + ' ' + idx_to_str[idx]
        # attempt a solve
        attempt_solve(net, cube, time_limit, stats, curr_scramble, i)

def attempt_solve(net, cube, time_limit, stats, scramble, solve_num=-1):
    '''
        attempts to solve the cube within a time limit using MCTS

        cube: the Cube object trying to be solved
        net: a CubeNet used to aid the tree traversal
        n: the number of traversals allowed in the attempt
        stats: a map keeping track of number of successful solves and solve attempts
    '''
    stats = stats if stats else { 'hits': 0, 'total': 0, 'time': 0, 'max': -1, 'fails': [] }
    tree = Tree(cube, net)
    start_time = time()
    print('attempting', scramble, '...', end='\r')
    while(time() - start_time < time_limit):
        leaf = traverse(tree.root)
        if(leaf.cube.is_solved()):
            solve_time = time() - start_time
            stats['hits'] += 1
            stats['time'] += solve_time
            stats['max'] = solve_time if solve_time > stats['max'] else stats['max']
            length = len(scramble.split(' ')) * 3 - 1
            if(solve_num >= 0):
                print('{0:>5}'.format(str(solve_num)+':'), end=' ')
            print(('solved {0:<'+str(length)+'}').format(scramble), '=>  ', ('{0:<'+str(length)+'}').format(get_solution(leaf, '')), str(round(solve_time, 2)), '\t', tree.root.N)
            break
        else:
            expand(leaf)
            update_statistics(leaf, get_value(leaf))
    else:
        print('failed solve', scramble, '\t', tree.root.N)
        stats['fails'].append(scramble)

    stats['total'] += 1

def get_solution(node, cur_sol):
    '''
        temp function for testing
        returns the solution found by mcts

        node: the node corresponding to the current cube state
        cur_sol: the string representation of the solution
    '''
    if(len(node.parents) == 0):
        return cur_sol
    parent = node.chosen_parent
    cur_sol = idx_to_str[parent.chosen_action] + ' ' + cur_sol
    return get_solution(parent, cur_sol)

def traverse(node):
    '''
        traverse the tree until you find a leaf
        add all of the leaf's children to the tree
        estimate value of leaf and update statistics

        node: a node to start the traversal from
    '''
    while(node.is_fully_expanded and not node.is_terminal):
        next_node = tree_policy(node)
        next_node.chosen_parent = node
        node = next_node
    return node

def tree_policy(node):
    '''
        chooses the next step in the tree traversal
        based on the tree policy, defined on page 5
        of this paper: https://arxiv.org/abs/1805.07470

        node: the node from which to pick the next step in the traversal
    '''
    # find best action
    best_a = None
    best_val = float('-inf')

    # summation term
    sum = 0
    for a_i in range(12):
        sum += node.N[a_i]

    # perform argmax on the actions
    for a in range(12):
        # this is the paper's tree policy
        # U term
        U = HYP_C * node.P[a].item() * sqrt(sum) / (1 + node.N[a])
        # Q term
        Q = node.W[a] - node.L[a]

        # this is slightly modified UCT
        # exploitation term
        exploit = node.W[a] + node.P[a].item()
        # exploration term
        explore = (HYP_UCT * math.log(sum + 1) / (node.N[a] + 1)) ** 0.5

        val = exploit + explore # or U + Q for paper's tree policy

        # check if new best
        if(val > best_val):
            best_a = a
            best_val = val
    # increase virtual loss to discourage identical traversal
    node.L[best_a] += HYP_V
    # remember chosen action
    node.chosen_action = best_a
    # return chosen child
    return node.children[best_a]


def expand(node):
    '''
        expands the given node by adding all of its children
        to the tree

        node: the node to be expanded
    '''
    for idx in range(12):
        next_cube = C.Cube(node.cube)
        next_cube.idx_turn(idx)
        next_node = None
        if(next_cube in node.map):
            next_node = node.map[next_cube]
        else:
            next_node = Node(next_cube, node.map, node.net)
            node.map[next_cube] = next_node

        node.children[idx] = next_node
        next_node.parents.append(node)
    node.is_fully_expanded = True

def get_value(node):
    '''
        calculate and return the value of the given
        node based on the trained network

        node: the node to be evaluated
    '''
    node.net.eval()
    with torch.no_grad():
        out_v, out_p = node.net(node.cube.to_tensor().to(device))
    return out_v.item()

def update_statistics(node, value):
    '''
        traverse up the tree, updating the statistics
        along the way

        node: the current stage of traversal
        value: the value of the leaf that's being propagated upward
    '''
    if(len(node.parents) == 0):
        return

    parent = node.chosen_parent
    action = parent.chosen_action

    node.chosen_parent = None
    parent.chosen_action = None

    parent.W[action] = max(parent.W[action], value)
    parent.N[action] += 1
    parent.L[action] -= HYP_V

    update_statistics(parent, value)
