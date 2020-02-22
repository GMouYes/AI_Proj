from Rules import *
from InstanceTemplate import *
import copy
import numpy as np
import math
import time
import random
import heapq


def generateZones(m: int, n: int, maxi: int, maxc: int, maxr: int):
    num_of_i = np.random.randint(0, maxi + 1)
    num_of_c = np.random.randint(0, maxc + 1)
    num_of_r = np.random.randint(0, maxr + 1)
    zones = []
    zone_labels = ["C"] * num_of_c + ["I"] * num_of_i + ["R"] * num_of_r
    random.shuffle(zone_labels)
    for label in zone_labels:
        location = (np.random.randint(0, m), np.random.randint(0, n))
        zone = createZone(label, location)
        zones.append(zone)
    return zones


def crossover(father: list, mother: list, n: int):
    cutpoint = np.random.randint(0, n)
    child1, child2 = [], []
    for zone in father:
        if zone.location[1] <= cutpoint:
            child1.append(zone)
        else:
            child2.append(zone)

    for zone in mother:
        if zone.location[1] > cutpoint:
            child1.append(zone)
        else:
            child2.append(zone)
    return child1, child2


def mutation(zones: list, m: int, n: int, num_of_mutation: int):
    if len(zones) == 0:
        return zones
    mutation_indices = random.sample(range(len(zones)), num_of_mutation)
    for index in mutation_indices:
        random_location = (np.random.randint(0, m), np.random.randint(0, n))
        zones[index] = createZone(zones[index].name, random_location)
    return zones


def genetic(urbanmap: Map, k1: int, k2: int, k3: int, max_iteration: int):
    maxi, maxc, maxr = urbanmap.maxIndustrial, urbanmap.maxCommercial, urbanmap.maxResidential
    m, n = np.shape(urbanmap.mapState)
    prev_best = float('-inf')
    population = []
    count = 0

    start_time = time.time()
    # initialize population
    while len(population) < k1:
        zones = generateZones(m, n, maxi, maxc, maxr)
        if ifValidZoneList(zones, urbanmap):
            population.append((urbanmap.get_score(zones), zones))

    # start genetic iteration
    while count <= max_iteration:
        parents = heapq.nlargest(k1 - k3, population)
        population = heapq.nlargest(k2, population)
        while len(population) < k1:
            father = random.choices(population=parents, k=1)[0]
            mother = random.choices(population=parents, k=1)[0]
            while father[1] == mother[1]:
                mother = random.choices(population=parents, k=1)[0]
            child1, child2 = crossover(father[1], mother[1], n)
            num_of_mutation = 1
            child1 = mutation(child1, m, n, num_of_mutation)
            child2 = mutation(child2, m, n, num_of_mutation)
            if ifValidZoneList(child1, urbanmap):
                population.append((urbanmap.get_score(child1), child1))
            if ifValidZoneList(child2, urbanmap) and len(population) < k1:
                population.append((urbanmap.get_score(child2), child2))
        if max(population)[0] > prev_best:
            count = 0
        elif max(population)[0] <= prev_best:
            count += 1
        prev_best = max(population)[0]

    end_time = time.time()

    return prev_best, max(population)[1], end_time-start_time


class MoveList(object):
    def __init__(self, state):
        self.start_state = state
        self.final_state = state
        self.moves = []
        super(MoveList, self).__init__()

    def score(self):
        return self.moves[-1]["score"] if len(self.moves) > 0 else -np.inf

    def final_score_time(self):
        for move in self.moves:
            if move["score"] == self.score():
                return move["elapsed_time"]

    def num_nodes_expanded(self):
        return sum(move["nodes_expanded"] for move in self.moves) if len(self.moves) > 0 else 0

    def effective_branching_factor(self):
        if len(self.moves) == 0:
            return 0
        else:
            nodes_expanded = np.array([move["nodes_expanded"] for move in self.moves])
            return np.mean(nodes_expanded[1:] / nodes_expanded[:-1])


class Annealer(object):
    def __init__(self, initial_temp):
        self.initial_temp = initial_temp
        self.temp = initial_temp
        super(Annealer, self).__init__()

    def cooling_schedule_log(self, timestep, base=np.e):
        self.temp = self.initial_temp / math.log(timestep + base, base)

    def cooling_schedule_geometric(self, timestep, ratio=0.7):
        if timestep > 0:
            self.temp *= ratio

    def jump_probability(self, old_val, new_val, cooling_schedule, timestep, free_param):
        if new_val >= old_val:
            jump_prob = 1
        else:
            jump_prob = math.exp((new_val - old_val) / self.temp)
        cooling_schedule(timestep, free_param)
        return jump_prob


def greedyHillClimb(start_map: Map, deadline=10, confidence_thresh=40,
                    max_sideways_moves=20, initial_temp=40, cooling_schedule="log", cooling_param=math.e):
    start_map_copy = copy.copy(start_map)
    # Do some initial setup here
    start_time = time.time()
    elapsed_time = 0
    cur_confidence = 0
    annealer = Annealer(initial_temp)
    cooling_func = annealer.cooling_schedule_log if cooling_schedule == "log" else annealer.cooling_schedule_geometric
    best_solution = MoveList(copy.copy(start_map.mapState))
    start_score = 0  # No score without zones
    nodes_expanded_total = 0
    branching_factors = []

    # While we still have time and aren't confident that we found the optimal solution, keep trying
    while elapsed_time < deadline and cur_confidence < confidence_thresh:
        cur_solution = MoveList(best_solution.start_state)
        cur_score = start_score
        start_map.mapState = copy.copy(best_solution.start_state)
        num_sideways_moves = 0
        num_iterations = 0
        zone_list = []

        # Two possible scenarios for us to stop: we run out of time, or we get stuck
        while elapsed_time < deadline and num_sideways_moves <= max_sideways_moves:
            # Get available moves from the current state
            neighbors = Rules.get_neighbors(zone_list, start_map)

            # Get min-valued successors here
            # min_hval = min(neighbor["function_value"] for neighbor in neighbors)
            # candidate_moves = [move for _, move in enumerate(neighbors) if move["function_value"] == min_hval]

            # Choose a candidate
            choice = random.choice(neighbors)

            # Make a decision based on simulated annealing
            jump_prob = annealer.jump_probability(cur_score, choice["score"], cooling_func, num_iterations,
                                                  cooling_param)
            jump = random.choices([True, False], weights=[jump_prob, 1 - jump_prob])[0]

            if jump:
                if cur_score == choice["score"]:
                    num_sideways_moves += 1
                else:
                    num_sideways_moves = 0
                cur_score = choice["score"]
                choice["nodes_expanded"] = len(neighbors)
                nodes_expanded_total += len(neighbors)
                cur_solution.moves.append(choice)
            # If we didn't jump stay where we are. Eventually we will go somewhere worse, or stop in the current state
            num_iterations += 1
            elapsed_time = time.time() - start_time
            choice["elapsed_time"] = elapsed_time

        # We stopped! Check if our solution is better than the current best.
        if cur_solution.score() > best_solution.score():
            cur_solution.final_state = start_map.mapState
            best_solution = cur_solution

        # If not, we become more confident our current best is the global optimum
        else:
            cur_confidence += 1
        branching_factors.append(cur_solution.effective_branching_factor())

    # Done searching; generate dictionary of results
    search_results = {
        "initMap": start_map_copy,
        "expandNodeCount": nodes_expanded_total,
        "elapsedTime": elapsed_time,
        "branchingFactor": np.mean(branching_factors),
        "score": best_solution.score(),
        "timeToBest": best_solution.final_score_time()
    }
    if len(best_solution.moves) > 0:
        for zone in best_solution.moves[-1]["zone_list"]:
            start_map.mapState[zone.location[0], zone.location[1]] = zone.name
    search_results["finalMap"] = start_map.mapState
    return search_results


