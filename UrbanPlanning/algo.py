from Rules import *
from InstanceTemplate import *
import copy
import numpy as np
import random
import heapq


def generateZones(m: int, n: int, maxi: int, maxc: int, maxr: int):
    num_of_i = np.random.randint(0, maxi+1)
    num_of_c = np.random.randint(0, maxc+1)
    num_of_r = np.random.randint(0, maxr+1)
    zones = []
    for i in range(num_of_i):
        location = (np.random.randint(0, m), np.random.randint(0, n))
        zone = createZone("I", location)
        zones.append(zone)
    for i in range(num_of_c):
        location = (np.random.randint(0, m), np.random.randint(0, n))
        zone = createZone("C", location)
        zones.append(zone)
    for i in range(num_of_r):
        location = (np.random.randint(0, m), np.random.randint(0, n))
        zone = createZone("R", location)
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
    mutation_indices = random.sample(range(len(zones)), num_of_mutation)
    for index in mutation_indices:
        random_location = (np.random.randint(0, m), np.random.randint(0, n))
        zones[index] = createZone(zones[index].name, random_location)
    return zones



def genetic(urbanmap: Map, k1: int, k2: int, k3: int):
    maxi, maxc, maxr = urbanmap.maxIndustrial, urbanmap.maxCommercial, urbanmap.maxResidential
    m, n = np.shape(urbanmap.mapState)
    prev_best = float('-inf')
    population = []
    count = 0

    # initialize population
    while len(population) < k1:
        zones = generateZones(m, n, maxi, maxc, maxr)
        if ifValidZoneList(zones):
            population.append((get_score(zones), zones))

    # start genetic iteration
    while count <= 5:
        parents = heapq.nlargest(k1 - k3, population)
        population = heapq.nlargest(k2, population)
        while len(population) < k1:
            father = random.choice(parents, cum_weights=[parent[0] for parent in parents])
            mother = random.choice(parents, cum_weights=[parent[0] for parent in parents])
            while father[1] == mother[1]:
                mother = random.choice(parents, cum_weights=[parent[0] for parent in parents])
            child1, child2 = crossover(father[1], mother[1], n)
            num_of_mutation = 1
            child1 = mutation(child1, m, n, num_of_mutation)
            child2 = mutation(child2, m, n, num_of_mutation)
            if ifValidZoneList(child1):
                population.append((get_score(child1), child1))
            if ifValidZoneList(child2) and len(population) < k1:
                population.append((get_score(child2), child2))
        if max(population)[0] > prev_best:
            count = 0
        elif max(population)[0] <= prev_best:
            count += 1
        prev_best = max(population)[0]

    return prev_best



