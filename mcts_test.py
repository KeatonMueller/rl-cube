import torch
import random, math
from math import sqrt

import cube as C

# exploration hyperparameter
HYP_C = 1.1
# virtual loss hyperparameter
HYP_V = 0.1

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
            out_v, out_p = net(cube.to_tensor())
        # prior probability of action a from this state
        self.P = out_p[0]

        self.cube = cube
        self.map = map
        self.net = net

        self.is_terminal = cube.is_solved()
        self.is_fully_expanded = self.is_terminal

def mcts_test(net, length, n):
    '''
        uses MCTS to test the network on every scramble of the requested length

        net: a CubeNet network
        length: the length of the scramble to attempt to solve
        n: the number of tree traversals allowed before abandoning the solve attempt
    '''
    stats = {
        'hits': 0,
        'total': 0
    }
    mcts_test_helper(net, C.Cube(), length, length, stats, n)
    print('mcts test: solved', stats['hits'], 'out of', stats['total'], '(' + str(round(stats['hits'] / stats['total'] * 100, 2)) + '%)', str(length) + '-move scrambles')

def mcts_test_helper(net, cube, curr_len, orig_len, stats, n):
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
                attempt_solve(net, C.Cube(cube), n, stats)
            else:
                # otherwise recurse and keep scrambling
                mcts_test_helper(net, cube, curr_len-1, orig_len, stats, n)
            # undo the turn
            cube.turn(face_, dir_, True)

def attempt_solve(net, cube, n, stats):
    '''
        attempts to solve the cube within a set number of
        tree traversals using MCTS

        cube: the Cube object trying to be solved
        net: a CubeNet used to aid the tree traversal
        n: the number of traversals allowed in the attempt
        stats: a map keeping track of number of successful solves and solve attempts
    '''
    tree = Tree(cube, net)
    solved = False
    for i in range(n):
        leaf = traverse(tree.root)
        if(leaf.cube.is_solved()):
            solved = True
            print('solution length:', get_length(leaf, 0))
            break
        else:
            expand(leaf)
            update_statistics(leaf, get_value(leaf))

    if(solved):
        stats['hits'] += 1
    stats['total'] += 1

def get_length(node, n):
    '''
        temp function for testing
        returns the length of the solution found by mcts

        node: the node corresponding to the solved cube state
    '''
    if(len(node.parents) == 0):
        return n
    return get_length(node.chosen_parent, n+1)

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
    # perform argmax on the actions
    best_a = None
    best_val = float('-inf')
    for a in range(12):
        # summation term
        sum = 0
        for a_i in range(12):
            sum += node.N[a_i]
        # U term
        U = HYP_C * node.P[a].item() * sqrt(sum) / (1 + node.N[a])
        # Q term
        Q = node.W[a] - node.L[a]
        # check if new best
        if(U + Q > best_val):
            best_a = a
            best_val = U + Q
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
        out_v, out_p = node.net(node.cube.to_tensor())
    return out_v

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
