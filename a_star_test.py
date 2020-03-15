import torch
from time import time
from heapq import heappush, heappop

import cube as C

# device for CPU or GPU calculations
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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

def a_star_test(model, length, time_limit):
    '''
        uses A* Search to test the network on scrambles of the requested length

        model: a ResCubeNet network
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
        a_star_test_helper_all(model, C.Cube(), length, length, stats, time_limit, '')
    # otherwise randomly scramble the cube `length` times
    else:
        a_star_test_helper_random(model, C.Cube(), length, stats, time_limit)

    # print out results of test
    print('A* test: solved', stats['hits'], 'out of', stats['total'], \
            '(' + str(round(stats['hits'] / stats['total'] * 100, 2)) + '%)',\
            str(length) + '-move scrambles')
    print('\tmax time: ' + str(round(stats['max'], 2)) + '\n' + \
          '\tavg time: ' + str(round(stats['time'] / stats['hits'], 2)))

    # record failed solves
    if(len(stats['fails']) > 0):
        with open('failed_solves.txt', 'w') as out_file:
            out_file.write(str(stats['fails']))


def a_star_test_helper_all(model, cube, curr_len, orig_len, stats, time_limit, curr_scramble):
    '''
        performs all 12 possible moves and either attempts a solve
        afterwards or recursively calls itself to further scramble the cube

        model: ResCubeNet network
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
            attempt_solve(model, C.Cube(cube), time_limit, stats, curr_scramble + ' ' + idx_to_str[idx])
        else:
            # otherwise recurse and keep scrambling
            a_star_test_helper_all(model, cube, curr_len-1, orig_len, stats, time_limit, curr_scramble + ' ' + idx_to_str[idx])
        # undo the turn
        cube.idx_turn(idx, True)

def a_star_test_helper_random(model, cube, length, stats, time_limit):
    '''
        randomly scrambles 1000 cubes `length` times and then attempts
        to solve them

        model: a ResCubeNet network
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
        attempt_solve(model, cube, time_limit, stats, curr_scramble, i)

def pop_batch(open_set, batch_size):
    '''
        pop a batch of cubes off of the open_set

        open_set: the open_set priority queue
        batch_size: the number of cubes to pop (must be <= len(open_set))
    '''
    popped = []
    for i in range(batch_size):
        popped.append(heappop(open_set)[1])
    return popped

def get_solution(cube, cur_sol, cube_to_parent):
    '''
        get the solution string found by A* search

        cube: the current cube state in the solution path
        cur_sol: the current solution string
        cube_to_parent: map from Cube object to parent Cube object
    '''
    if(cube_to_parent[cube] == None):
        return cur_sol
    parent_cube, idx = cube_to_parent[cube]
    cur_sol = idx_to_str[idx] + ' ' + cur_sol
    return get_solution(parent_cube, cur_sol, cube_to_parent)

def attempt_solve(model, start_cube, time_limit, stats, scramble, solve_num=-1):
    '''
        attempts to solve the given cube in the given time with the given model

        model: a ResCubeNet model used as the heuristic function
        start_cube: a Cube object to solve
        time_limit: the max time allowed to try and solve the cube
        stats: a map keeping track of number of successful solves and solve attempts
        scramble: a string holding the scramble being solved
        solve_num: (optional) the number solve being attempted
    '''
    stats = stats if stats else { 'hits': 0, 'total': 0, 'time': 0, 'max': -1, 'fails': [] }
    batch_size = 1
    start_time = time()
    open_set = []
    cube_to_path_cost = dict()
    cube_to_heuristic = dict()
    cube_to_f_score = dict()
    cube_to_parent = dict()

    model.eval()
    with torch.no_grad():
        heuristic = model(start_cube.to_tensor().to(device)).item()
        cube_to_path_cost[start_cube] = 0
        cube_to_heuristic[start_cube] = heuristic
        cube_to_f_score[start_cube] = heuristic
        cube_to_parent[start_cube] = None

        heappush(open_set, (cube_to_f_score[start_cube], start_cube))

        while(time() - start_time < time_limit and len(open_set) > 0):
            # get batch of cubes from open set
            cubes = pop_batch(open_set, min(batch_size, len(open_set)))
            cubes_to_compute = []
            # for each popped cube
            for cube in cubes:
                # if solved, we're done
                if(cube.is_solved()):
                    solve_time = time() - start_time
                    stats['hits'] += 1
                    stats['time'] += solve_time
                    stats['max'] = solve_time if solve_time > stats['max'] else stats['max']
                    stats['total'] += 1
                    length = len(scramble.split(' ')) * 3 - 1
                    if(solve_num >= 0):
                        print('{0:>5}'.format(str(solve_num)+':'), end=' ')
                    print(('solved {0:<'+str(length)+'}').format(scramble), '=>  ', ('{0:<'+str(length)+'}').format(get_solution(cube, '', cube_to_parent)), str(round(solve_time, 2)))
                    return
                # get all 12 neighboring Cube objects
                neighbors = C.get_neighbors(cube)
                # for each neighbor
                for idx, neighbor in enumerate(neighbors):
                    # if we've never seen this state before
                    if(neighbor not in cube_to_f_score):
                        # compute its heuristic later
                        cubes_to_compute.append(neighbor)
                        # set its path cost and parent
                        cube_to_path_cost[neighbor] = cube_to_path_cost[cube] + 1
                        cube_to_parent[neighbor] = (cube, idx)
                    # if we have seen this state
                    else:
                        # see if we just found a better path
                        tentative_path_cost = cube_to_path_cost[cube] + 1
                        if(tentative_path_cost < cube_to_path_cost[neighbor]):
                            cube_to_parent[neighbor] = (cube, idx)
                            cube_to_path_cost[neighbor] = tentative_path_cost
                            cube_to_f_score[neighbor] = tentative_path_cost + cube_to_heuristic[neighbor]
                            if((cube_to_f_score[neighbor], neighbor) not in open_set):
                                heappush(open_set, (cube_to_f_score[neighbor], neighbor))
            # input tensor to calculate heuristics
            inputs = torch.empty(len(cubes_to_compute), 480, device=device)
            # populate tensor
            for i, cube in enumerate(cubes_to_compute):
                inputs[i] = cube.to_tensor()
            # get heuristic calculations
            outputs = model(inputs)
            # update heuristic and f_scores
            for i, cube in enumerate(cubes_to_compute):
                cube_to_heuristic[cube] = outputs[i].item()
                cube_to_f_score[cube] = cube_to_path_cost[cube] + cube_to_heuristic[cube]
                heappush(open_set, (cube_to_f_score[cube], cube))
        print('failed solve', scramble)
        stats['fails'].append(scramble)
        stats['total'] += 1
