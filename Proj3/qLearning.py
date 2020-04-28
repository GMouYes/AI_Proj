'''
the actual q learning file
for the exact search algorithm
'''

# import numpy as np

from expSearch import *
import time
import numpy as np
import random


def search(**args):
    '''
    placeholder function
    # TODO: implement details
    '''
    world = args["world"]
    height, width = world.shape
    searchType = args["algorithm"]
    startPosition = args["startPosition"]
    moveCost = args["moveCost"]
    maxTime = args["maxTime"]
    transitionProb = args["transitionProb"]

    # expected return: dict of following
    policy = np.zeros((height, width))  # 0,1,2,3 indicating up, right, down, left
    reward = np.zeros((height, width))  # real values
    Q = np.zeros((height * width, 4))

    result = {
        "policy": policy,
        "reward": reward,
        "time": 0,
        # others to be determined
    }

    lookForPath(startPosition, world, moveCost, maxTime, 0, transitionProb)

    return result


def translateProcedure(stepCounter, position, chooseDirection, direction, newPosition):
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
    print("Reach end pos:", position)
    print("Final reward:", world[position])
    print("Overall reward:", reward)
    print("")


def lookForPath(startPosition, world, moveCost, maxTime, searchType, ratio):
    stepCounter = 0
    position = startPosition
    startTime = time.time()
    while True:
        if isEnd(world, position):
            reward = pathReward(world, position, stepCounter, moveCost)

            endState(position, world, reward)

            break

        if time.time() - startTime > maxTime:
            print("Max time reached, ending at step", stepCounter)
            break

        stepCounter += 1
        chooseDirection = policyDirection(searchType)
        direction = actualDirection(chooseDirection, ratio)
        newPosition = actualPosition(world, position, direction)

        translateProcedure(stepCounter, position, chooseDirection, direction, newPosition)

        position = newPosition

    return


def updatePolicy():
    pass


def updateReward():
    pass


def update(height, width, Q, s, a, s_new, r, lr=0.1, gamma=0.95):
    # index = convertPositiontoIndex(s, height, width)
    # new_index = convertPositiontoIndex(s_new, height, width)
    # Q[index, a] = Q[index, a] + lr * (r + gamma * np.max(Q[new_index]) - Q[index, a])
    Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[s_new]) - Q[s][a])
    return Q


def play(iteration, startPosition, Q, ratio, world, movecost):
    height, width = np.shape(world)
    for _ in range(iteration):
        # done = False
        position = startPosition
        stepCounter = 0
        while True:
            stepCounter += 1
            chooseDirection = policyDirection(1, Q, position)
            direction = actualDirection(chooseDirection, ratio)
            newPosition = actualPosition(world, position, direction)
            reward = giveReward(newPosition, world) + movecost
            Q = update(height, width, Q, position, direction, newPosition, reward)
            translateProcedure(stepCounter, position, chooseDirection, direction, newPosition)
            position = newPosition
            if isEnd(world, newPosition):
                break
        reward = pathReward(world, position, stepCounter, movecost)
        endState(position, world, reward)

    return Q




def pathReward(world, position, stepCounter, moveCost):
    finalReward = world[position]
    return finalReward + stepCounter * moveCost


def isEnd(world, position: tuple):
    if world[position] != 0.:
        return True
    return False


def policyDirection(searchType, Q, position):
    # return direction based on algorithm
    # still placeholder
    if searchType == 0:
        return random.randint(0, 3)
    if searchType == 1:
        # index = convertPositiontoIndex(position, height, width)
        # return np.argmax(Q[index])
        # return np.argmax(Q[position[0], position[1]])
        moves = Q[position]
        max_value = np.max(moves)
        # choices = []
        # for i in range(len(moves)):
        #     if moves[i] == max_value:
        #         choices.append(i)
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


def convertPositiontoIndex(position, height, width):
    return position[0] * height + width


def convertIndextoPosition(index, width):
    return index // width, index % width


def giveReward(position, world):
    return world[position[0], position[1]]

def main():
    world = np.zeros((4, 6))
    world[0, 0] = 1
    world[1, 0] = -1
    Q = np.zeros((4, 6, 4))
    res = play(iteration=1, startPosition=(3, 0), Q=Q, ratio=0.6, world=world, movecost=-0.04)
    print(res)

if __name__ == '__main__':
    main()