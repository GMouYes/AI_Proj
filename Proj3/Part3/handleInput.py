import sys
import numpy as np


def readFile(fileName):
    data = np.genfromtxt(fileName, delimiter=',')
    # print(data.shape, data[0][0], type(data[0][0]))

    return data


def formatError():
    ''' print out error and info to fix it '''
    print("-----------------------------------------------------------------------------")
    print("There's some problem with reading inputs, please check again:")
    print("You should have 4 explicit inputs.")
    print("1. Capacity of the truck, a positive integer.")
    print("2. Length of the road, a positive integer.")
    print("3. Penalty for starting the truck, natural number.")
    print("4. Number of clock ticks, where:")
    print("\t it is either a positive integer for the limitation;")
    print("\t or -1 for \"forever\" (a large enough number)")
    print("-----------------------------------------------------------------------------")
    return True


def repeatChoice(truckCapacity, lengthOfRoad, startingPenalty, maxClockTicks):
    ''' interpret command and print it out '''
    print("-------------------------------------------")
    print("Input processing complete, you have chosen:")
    print("-------------------------------------------")

    print("Truck capacity:\t\t", truckCapacity)
    # print("-------------------------------------------")

    print("Length of the road:\t", lengthOfRoad)
    # print("-------------------------------------------")

    print("Truck start penalty:\t", startingPenalty)
    print("-------------------------------------------")

    print("Number of clock ticks:\t", end=" ")
    if maxClockTicks == -1:
        print("up to us")
    else:
        print(maxClockTicks)
    print("-------------------------------------------")

    print("Now searching, this may take some time ...")
    # print("-------------------------------------------")
    return True


def readInput():
    ''' consume input from command line and handle them '''
    # check passing in parameters
    if len(sys.argv) < 5:
        formatError()
        return False

    truckCapacity = sys.argv[1]
    lengthOfRoad = sys.argv[2]
    startingPenalty = sys.argv[3]
    maxClockTicks = sys.argv[4]

    # check legal input strings
    try:
        truckCapacity = int(truckCapacity)
    except Exception as e:
        formatError()
        return False

    if truckCapacity <= 0:
        formatError()
        return False

    try:
        lengthOfRoad = int(lengthOfRoad)
    except Exception as e:
        formatError()
        return False

    if lengthOfRoad <= 0:
        formatError()
        return False

    try:
        startingPenalty = float(startingPenalty)
    except Exception as e:
        formatError()
        return False

    try:
        maxClockTicks = int(maxClockTicks)
    except Exception as e:
        formatError()
        return False

    if maxClockTicks <= 0 and maxClockTicks != -1:
        formatError()
        return False

    repeatChoice(truckCapacity, lengthOfRoad, startingPenalty, maxClockTicks)

    return truckCapacity, lengthOfRoad, startingPenalty, maxClockTicks
