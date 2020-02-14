from queue import PriorityQueue
from Board import *
import math
import numpy as np
import copy
import time
import random


def A_Star(start: board, h_type):
    # sth to implement here
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    cur_state = None

    start_time = time.time()

    while not frontier.empty():
        cur_state = frontier.get()[1]

        if cur_state.finished():
            break

        for next_state in cur_state.neighbors():
            new_cost = cost_so_far[cur_state] + next_state[1]
            if next_state[0] not in cost_so_far or new_cost < cost_so_far[next_state[0]]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + next_state[0].heuristic(h_type)
                frontier.put((priority, next_state))
                came_from[next_state[0]] = cur_state

    end_time = time.time()

    path = []
    while cur_state != start:
        path.append(cur_state)
        cur_state = came_from[cur_state]
    path.append(start)
    # please add the time elapsedtime to return statement
    return path.reverse()


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

    def jump_probability(self, old_val, new_val, cooling_schedule, timestep, free_param):
        if new_val <= old_val:
            jump_prob = 1
        else:
            jump_prob = math.exp((old_val - new_val)/self.temp)
        cooling_schedule(timestep, free_param)
        return jump_prob


def greedyHillClimb(start_board: board, h_type):
    # Do some initial setup here
    start_time = time.time()
    elapsed_time = 0
    deadline = 10  # Seconds before we run out of time

    confidence_thresh = 100  # Alternatively, we terminate if we fail to find a better solution in confidence_thresh
                             # tries.

    max_sideways_moves = 100  # For a given iteration, how many sideways moves are allowed before we accept the
                              # current solution.
    cur_confidence = 0
    initial_temp = 30  # Starting temp for simulated annealing
    best_solution = MoveList(copy.deepcopy(start_board.state))
    cur_hval = start_board.heuristic(h_type)  # Heuristic function value of the start state

    # While we still have time and aren't confident that we found the optimal solution, keep trying
    while elapsed_time < deadline and cur_confidence < confidence_thresh:
        cur_solution = MoveList(best_solution.start_state)
        start_board.state = copy.deepcopy(best_solution.start_state)
        num_sideways_moves = 0
        annealer = Annealer(initial_temp)
        num_iterations = 0

        # Three possible scenarios for us to stop: we run out of time, we find a solution, or we get stuck
        while elapsed_time < deadline and cur_hval > 0 and num_sideways_moves < max_sideways_moves:
            # Get available moves from the current state
            neighbors = start_board.get_neighbors_in_place(h_type)

            # Get min-valued successors here
            min_hval = min(neighbor["function_value"] for neighbor in neighbors)
            candidate_moves = [move for _, move in enumerate(neighbors) if move["function_value"] == min_hval]

            # Choose a candidate
            choice = random.choice(candidate_moves)

            # Make a decision based on simulated annealing
            jump_prob = annealer.jump_probability(cur_hval, min_hval, annealer.cooling_schedule_log, num_iterations,
                                                  math.e)
            jump = random.choices([True, False], weights=[jump_prob, 1 - jump_prob])[0]

            if jump:
                if cur_hval == min_hval:
                    num_sideways_moves += 1
                else:
                    num_sideways_moves = 0
                cur_hval = min_hval
                choice["nodes_expanded"] = len(neighbors)
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

    # Done searching; generate dictionary of results
    start_board.state = copy.deepcopy(best_solution.start_state)
    search_results = {
        "initBoard": start_board,
        "expandNodeCount": best_solution.num_nodes_expanded(),
        "elapsedTime": best_solution.moves[-1]["elapsed_time"],
        "branchingFactor": best_solution.effective_branching_factor(),
        "cost": best_solution.cost(),
        "sequence": best_solution.moves
    }
    return search_results
