import sys
from HandleOutput import *

import numpy as np


def readFile(fileName):
    with open(fileName, "r") as f:
        info = [line.strip("\n") for line in f]

    industrial, commercial, residential = [int(item) for item in info[:3]]
    siteMap = np.array([list(row.split(",")) for row in info[3:]])

    return industrial, commercial, residential, siteMap


def formatError():
    print("-----------------------------------------------------------------------------")
    print("There's some problem with reading inputs, please check again:")
    print("You should have 2 explicit inputs.")
    print("1. Initial map state, indicated by the complete path of a file.")
    print("2. Type of searching algo, \"HC\" for Hill Climbing, \"GA\" for Genetic Algorithm, or \"SGHC\" for "
          "Super-Greedy Hill Climbing")
    print("-----------------------------------------------------------------------------")
    return True


def repeatChoice(industrial, commercial, residential, siteMap, searchType):
    print("-------------------------------------------")
    print("Input processing complete, you have chosen:")
    print("-------------------------------------------")

    print("Max industrial zones:", industrial)
    print("Max commercial zones:", commercial)
    print("MAx residential zones:", residential)
    print("-------------------------------------------")

    print("Map State:")
    print(siteMap)
    # printMap(siteMap)
    print("-------------------------------------------")

    print("Searching Algo:", end=" ")
    if searchType == "hc":
        print("Hill Climbing")
    elif searchType == "ga":
        print("Genetic Algorithm")
    else:
        print("Super-Greedy Hill Climbing")
    # print("-------------------------------------------")

    print("Now searching, this may take some time ...")
    print("-------------------------------------------")
    return True


def readInput():
    # check passing in parameters
    if len(sys.argv) < 3:
        formatError()
        return False

    fileName = sys.argv[1]
    searchType = sys.argv[2].lower()

    # check legal input strings
    try:
        industrial, commercial, residential, siteMap = readFile(fileName)
    except Exception as e:
        formatError()
        return False

    if searchType not in ["hc", "ga", "sghc"]:
        formatError()
        return False

    repeatChoice(industrial, commercial, residential, siteMap, searchType)

    return industrial, commercial, residential, siteMap, searchType
