from HandleInput import *
from HandleOutput import *
from Board import *
from searchAlgo import *
import time


def search(initBoard, searchType, heuristicFunc):
    searchResults = {
        "initBoard": None,
        "expandNodeCount": None,
        "elapsedTime": None,
        "branchingFactor": None,
        "cost": None,
        "sequence": None,
    }

    if searchType == "2":
        return greedyHillClimb(initBoard, heuristicFunc)
    else:
        return A_Star(initBoard,heuristicFunc)


def main():
    try:
        initBoard, searchType, heuristicFunc = readInput()
    except Exception as e:
        return False

    # broadcast what we read from the input
    repeatChoice(initBoard, searchType, heuristicFunc)

    # please, make your returning searchResults a dic
    searchResults = search(initBoard,searchType,heuristicFunc)
    #print(searchResults)
    generateOutput(searchResults)
    return True


if __name__ == '__main__':
    main()
