from handleOutput import *
from expSearch import *

def readFile(fileName):
    data = np.genfromtxt(fileName, delimiter=',')
    #print(data.shape, data[0][0], type(data[0][0]))

    return data

def formatError():
    ''' print out error and info to fix it '''
    print("-----------------------------------------------------------------------------")
    print("There's some problem with reading inputs, please check again:")
    print("You should have 3 explicit inputs.")
    print("1. Initial data map/board, indicated by the complete path of a file.")
    print("2. Movement cost for each step.")
    print("3. Probability of same direction movement. Some value between 0 and 1.")
    print("-----------------------------------------------------------------------------")
    return True


def repeatChoice(fileName, data, moveCost, transitionProb):
    ''' interpret command and print it out '''
    print("-------------------------------------------")
    print("Input processing complete, you have chosen:")
    print("-------------------------------------------")

    print("Target file to read:\t\t", fileName)
    # print("-------------------------------------------")
    

    print("Movement cost for each step:\t", moveCost)
    #print("-------------------------------------------")

    print("Same direction probability:\t", transitionProb)
    print("-------------------------------------------")

    print("Grid world we got:")
    print(data)
    print("-------------------------------------------")

    print("Now searching, this may take some time ...")
    # print("-------------------------------------------")
    return True


def readInput():
    ''' consume input from command line and handle them '''
    # check passing in parameters
    if len(sys.argv) < 4:
        formatError()
        return False

    fileName = sys.argv[1]
    moveCost = sys.argv[2]
    transitionProb = sys.argv[3]

    # check legal input strings
    try:
        data = readFile(fileName)
    except Exception as e:
        print("Problem with reading the data points, please try again.")
        return False

    try:
        moveCost = float(moveCost)
    except Exception as e:
        formatError()
        return False

    try:
        transitionProb = float(transitionProb)
    except Exception as e:
        formatError()
        return False

    if transitionProb < 0 or transitionProb > 1:
        formatError()
        return False

    repeatChoice(fileName, data, moveCost, transitionProb)

    return data, moveCost, transitionProb