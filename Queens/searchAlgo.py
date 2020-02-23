from queue import PriorityQueue
from Board import *
import math
import numpy as np
import copy
import time
import random


def tostring(state, m, n):
    string = ""
    for i in range(n):
        for j in range(m):
            if state[j, i] != 0:
                string += str(j)
                if i < n-1:
                    string += ','
    return string


def toboard(string, weight, m, n):
    pos = string.split(",")
    # weight = [81, 9, 1, 16, 4]
    new_board = np.zeros((m, n), int)
    for i in range(n):
        new_board[int(pos[i]), i] = weight[i]
    return new_board


def A_Star(start: board, h_type: str):
    # get the shape of board
    m, n = np.shape(start.state)
    # get the weight of each queen
    weight = []
    for i in range(n):
        for j in range(m):
            if start.state[j, i] != 0:
                weight.append(start.state[j, i])

    frontier = PriorityQueue()
    frontier.put(start)
    came_from = dict()
    cost_so_far = dict()
    came_from[tostring(start.state, m, n)] = None
    cost_so_far[tostring(start.state, m, n)] = 0
    path = []
    nodes_expanded = 0
    result = dict()
    result["initBoard"] = start
    result["branchingFactor"] = 1

    start_time = time.time()

    while not frontier.empty():
        cur_state = frontier.get()
        cur_string = tostring(cur_state.state, m, n)
        cur_time = time.time()

        if cur_state.heuristic(h_type) == 0:
            result["cost"] = cost_so_far[cur_string]
            while not cur_string == tostring(start.state, m, n):
                path.append(toboard(cur_string, weight, m, n))
                cur_string = came_from[cur_string]
            path.append(start.state)
            path.reverse()
            result["elapsedTime"] = cur_time - start_time
            result["sequence"] = path
            result["expandNodeCount"] = nodes_expanded
            break

        for next_state in cur_state.get_neighbors():
            new_cost = cost_so_far[cur_string] + next_state[1]
            state_string = tostring(next_state[0].state, m, n)
            if state_string not in cost_so_far or new_cost < cost_so_far[state_string]:
                nodes_expanded += 1
                cost_so_far[state_string] = new_cost
                next_state[0].priority = new_cost + next_state[0].heuristic(h_type)
                frontier.put(next_state[0])
                came_from[state_string] = tostring(cur_state.state, m, n)

    return result


class MoveList(object):
    def __init__(self, state):
        self.start_state = state
        self.final_state = state
        self.moves = []
        super(MoveList, self).__init__()

    def cost(self):
        return sum(move["move_cost"] for move in self.moves) if len(self.moves) > 0 else math.inf

    def num_nodes_expanded(self):
        return sum(move["nodes_expanded"] for move in self.moves) if len(self.moves) > 0 else 0

    def effective_branching_factor(self):
        if len(self.moves) == 0:
            return 0
        else:
            nodes_expanded = np.array([move["nodes_expanded"] for move in self.moves])
            return np.mean(nodes_expanded[1:]/nodes_expanded[:-1])


class Annealer(object):
    def __init__(self, initial_temp):
        self.initial_temp = initial_temp
        self.temp = initial_temp
        super(Annealer, self).__init__()

    def cooling_schedule_log(self, timestep, base=np.e):
        self.temp = self.initial_temp/math.log(timestep + base, base)

    def cooling_schedule_geometric(self, timestep, ratio=0.7):
        if timestep > 0:
            self.temp *= ratio

    def jump_probability(self, old_val, new_val, cooling_schedule, timestep, free_param):
        if new_val <= old_val:
            jump_prob = 1
        else:
            jump_prob = math.exp((old_val - new_val)/self.temp)
        cooling_schedule(timestep, free_param)
        return jump_prob


def greedyHillClimb(start_board: board, h_type, mode="normal", deadline=10, confidence_thresh=100,
                    max_sideways_moves=100, initial_temp=80, cooling_schedule="geom", cooling_param=0.4):
    start_board_copy = copy.copy(start_board)
    # Do some initial setup here
    start_time = time.time()
    elapsed_time = 0
    cur_confidence = 0
    annealer = Annealer(initial_temp)
    cooling_func = annealer.cooling_schedule_log if cooling_schedule == "log" else annealer.cooling_schedule_geometric
    best_solution = MoveList(copy.copy(start_board.state))
    start_hval = start_board.heuristic(h_type)  # Heuristic function value of the start state
    nodes_expanded_total = 0
    branching_factors = []

    # While we still have time and aren't confident that we found the optimal solution, keep trying
    while elapsed_time < deadline and cur_confidence < confidence_thresh:
        cur_solution = MoveList(best_solution.start_state)
        cur_hval = start_hval
        start_board.state = copy.copy(best_solution.start_state)
        num_sideways_moves = 0
        num_iterations = 0

        # Three possible scenarios for us to stop: we run out of time, we find a solution, or we get stuck
        while elapsed_time < deadline and cur_hval > 0 and num_sideways_moves <= max_sideways_moves:
            # Get available moves from the current state
            neighbors = start_board.get_neighbors_in_place(h_type)

            if mode == "super_greedy":
                # Get min-valued successors here
                new_hval = min(neighbor["function_value"] for neighbor in neighbors)
                candidate_moves = [move for _, move in enumerate(neighbors) if move["function_value"] == new_hval]
            else:
                candidate_moves = neighbors

            # Choose a candidate
            choice = random.choice(candidate_moves)

            # Make a decision based on simulated annealing
            jump_prob = annealer.jump_probability(cur_hval, choice["function_value"], cooling_func, num_iterations,
                                                  cooling_param)
            jump = True if random.random() <= jump_prob else False

            if jump:
                if cur_hval == choice["function_value"]:
                    num_sideways_moves += 1
                else:
                    num_sideways_moves = 0
                cur_hval = choice["function_value"]
                choice["nodes_expanded"] = len(neighbors)
                nodes_expanded_total += len(neighbors)
                cur_solution.moves.append(choice)
                start_board.make_move(choice["col"], choice["end_pos"])
            # If we didn't jump stay where we are. Eventually we will go somewhere worse, or stop in the current state
            num_iterations += 1
            elapsed_time = time.time() - start_time
            choice["elapsed_time"] = elapsed_time

        # We stopped! Check if our solution is better than the current best.
        if cur_solution.cost() < best_solution.cost():
            cur_solution.final_state = start_board.state
            best_solution = cur_solution

        # If not, we become more confident our current best is the global optimum
        else:
            cur_confidence += 1
        branching_factors.append(cur_solution.effective_branching_factor())

    # Done searching; generate dictionary of results
    search_results = {
        "initBoard": start_board_copy,
        "expandNodeCount": nodes_expanded_total,
        "elapsedTime": elapsed_time,
        "bestSolutionTime": best_solution.moves[-1]["elapsed_time"] if len(best_solution.moves) > 0 else np.inf,
        "branchingFactor": np.mean(branching_factors),
        "cost": best_solution.cost(),
        "solved": len(best_solution.moves) > 0 and best_solution.moves[-1]["function_value"] == 0
    }
    move_states = []
    for move in best_solution.moves:
        start_board.make_move(move["col"], move["end_pos"])
        move_states.append(copy.copy(start_board.state))
    search_results["sequence"] = move_states
    return search_results



