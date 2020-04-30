'''
the actual q learning file
for the exact search algorithm
'''

from expSearch import *
import time
import numpy as np
import random
import collections
import itertools


def search(**args):
    '''
    '''
    world = args["world"]
    height, width = world.shape
    # searchType = kwargs["algorithm"]
    startPosition = args["startPosition"]
    moveCost = args["moveCost"]
    maxTime = args["maxTime"]
    transitionProb = args["transitionProb"]
    threshold = args["tol"]
    maxMoves = args["maxMoves"]

    # policy = np.zeros((height, width))  # 0,1,2,3 indicating up, right, down, left
    # reward = np.zeros((height, width))  # real values
    Q = np.zeros((height, width, 4))
    Q, policy, time, epoch, reward = play(startPosition=startPosition, Q=Q, ratio=transitionProb, world=world, movecost=moveCost,
                           maxtime=maxTime, threshold=threshold, maxMoves=maxMoves)

    # expected return: dict of following
    result = {
        "policy": policy,
        "Q": Q,
        "time": time,
        "epoch": epoch,
        "reward": reward,
        # others to be determined
    }

    return result


def translateProcedure(stepCounter, position, chooseDirection, direction, newPosition):
    # print for debugging at each iter
    directionDict = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    print("Step", stepCounter)
    print("startPosition:", position)
    print("Choosen direction:", directionDict[chooseDirection])
    print("Actual direction:", directionDict[direction])
    print("Arrival position:", newPosition)
    print("")
    return


def endState(position, world, reward):
    # print for debugging at each epoch
    print("Reach end pos:", position)
    print("Final reward:", world[position])
    print("Overall reward:", reward)
    print("")


def update(Q, s, a, s_new, r, lr=0.1, gamma=0.95):
    Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[s_new]) - Q[s][a])
    return Q


def play(startPosition, Q, ratio, world, movecost, maxtime, threshold, maxMoves):
    #height, width = np.shape(world)
    startTime = time.time()
    epoch = 0
    reward = 0
    prev_policies = np.empty((world.shape[0], world.shape[1], 100))

    while True:
        epoch += 1
        # termination
        t = time.time()-startTime
        if t > maxtime-threshold:
            break

        if epoch > 100:
            # Convergence criterion
            diffs = (prev_policies[:, :, 1:100] - prev_policies[:, :, 0:99])
            diffs = np.abs(diffs) >= np.finfo(np.float).eps
            diff_pct = np.sum(diffs, axis=(0, 1)) / (world.shape[0] * world.shape[1])
            if np.all(diff_pct <= 0.01):
                break

        position = startPosition
        stepCounter = 0

        while True:
            stepCounter += 1
            # choose dir according to policy
            chooseDirection = policyDirection(1, Q, position)
            # actual dir according to transition
            direction = actualDirection(chooseDirection, ratio)
            # the expected landing position
            newPosition = actualPosition(world, position, direction)
            # update reward and Q value
            reward = giveReward(newPosition, world) + movecost
            Q = update(Q, position, chooseDirection, newPosition, reward)
            # debug print
            # translateProcedure(stepCounter, position, chooseDirection, direction, newPosition)
            # update new position
            position = newPosition
            # ending criteria
            if isEnd(world, newPosition) or stepCounter >= maxMoves:
                break
        # final reward
        reward = pathReward(world, position, stepCounter, movecost)
        prev_policies[:, :, 0:99] = prev_policies[:, :, 1:100]
        prev_policies[:, :, 99] = np.argmax(Q, axis=-1)
        # debug print
        # endState(position, world, reward)

    policy = np.argmax(Q, axis=-1)
    expected_reward = np.max(Q[startPosition])

    return Q, policy, t, epoch, expected_reward


def pathReward(world, position, stepCounter, moveCost):
    finalReward = world[position]
    return finalReward + stepCounter * moveCost


def isEnd(world, position: tuple):
    return True if world[position] != 0. else False


def policyDirection(searchType, Q, position):
    # return direction based on algorithm
    if searchType == 0:
        return random.randint(0, 3)
    if searchType == 1:
        moves = Q[position]
        max_value = np.max(moves)
        choices = [index for index,value in enumerate(moves) if value == max_value]
        return np.random.choice(choices, 1)[0]
    return 0


def actualDirection(direction, ratio):
    guess = random.random()
    if guess < ratio:
        return direction

    if guess < (1. + ratio) / 2.:
        return (direction - 1) % 4

    return (direction + 1) % 4


def actualPosition(world, position, direction):
    y, x = position
    height, width = world.shape

    if direction == 0:
        # up
        if y == 0:
            return (y, x)
        return (y - 1, x)

    if direction == 1:
        # right
        if x == width - 1:
            return (y, x)
        return (y, x + 1)

    if direction == 2:
        # down
        if y == height - 1:
            return (y, x)
        return (y + 1, x)

    if direction == 3:
        # left
        if x == 0:
            return (y, x)
        return (y, x - 1)


def giveReward(position, world):
    return world[position]
