from time import time
from heapq import heappush, heappop

import cube as C

class Node:
    def __init__(state, parent, path_cost, heuristic):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.total_cost = self.path_cost + self.heuristic

def a_star_test(model, length, time_limit):
    pass

def pop_batch(open_set, batch_size):
    popped = []
    for i in range(batch_size):
        popped.append(heappop(open_set)[1])
    return popped

def attempt_solve(model, cube, time_limit, stats, scramble, solve_num=-1):
    batch_size = 1
    start_time = time()
    open_set = []
    cube_to_node = map()
    model.eval()
    with torch.no_grad():
        heuristic = model(cube.to_tensor().to(device)).item()

    node = Node(cube, None, 0, heuristic)
    heappush(open_set, (node.f_cost, node))
    cube_to_node[cube] = node

    while(time() - start_time < time_limit and len(open_set) > 0):
        nodes = pop_batch(open_set, min(batch_size, len(open_set)))
        for node in nodes:
            if(node.state.is_solved()):
                print('solved!')
                break
            neighbors = C.get_neighbors(node.state)
            
