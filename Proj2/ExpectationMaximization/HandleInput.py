import sys
from HandleOutput import *

import numpy as np

def readFile(fileName):
    data = np.genfromtxt(fileName, delimiter=',')
    #print(data.shape, data[0][0], type(data[0][0]))

    return data

def formatError():
    ''' print out error and info to fix it '''
    print("-----------------------------------------------------------------------------")
    print("There's some problem with reading inputs, please check again:")
    print("You should have 2 explicit inputs.")
    print("1. Initial data points, indicated by the complete path of a file.")
    print("2. Number of clusters to find. Typically, 0 means find the best one.")
    print("   Negative values are invalid.")
    print("-----------------------------------------------------------------------------")
    return True


def repeatChoice(fileName, data, clusters):
    ''' interpret command and print it out '''
    print("-------------------------------------------")
    print("Input processing complete, you have chosen:")
    print("-------------------------------------------")

    print("Target file to read:\t", fileName)
    #print("-------------------------------------------")

    print("Number of data points:\t", data.shape[0])
    #print("-------------------------------------------")

    print("Number of clusters:\t", end=" ")
    if clusters == 0:
        print("We choose the best")
    else:
        print(clusters)
    print("-------------------------------------------")

    print("Now searching, this may take some time ...")
    #print("-------------------------------------------")
    return True


def readInput():
    ''' consume input from command line and handle them '''
    # check passing in parameters
    if len(sys.argv) < 3:
        formatError()
        return False

    fileName = sys.argv[1]
    clusters = sys.argv[2]

    # check legal input strings
    try:
        data = readFile(fileName)
    except Exception as e:
        print("Problem with reading the data points, please try again.")
        return False

    try:
        clusters = int(clusters)
    except Exception as e:
        formatError()
        return False

    if clusters < 0:
        formatError()
        return False

    repeatChoice(fileName, data, clusters)

    return data, clusters