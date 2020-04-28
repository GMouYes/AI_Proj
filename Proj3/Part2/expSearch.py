'''
This is the file controlling all hypers
calling different components
and executing experiments
'''

# python3's own
import time
import numpy as np
import sys
import random

# we wrote them
from handleInput import *
from handleOutput import *
from qLearning import *

def hypers(data, moveCost, transitionProb):
	'''
	all hypers should be defined here
	returning the dict of them
	'''

	hyperDict = {
		# if you need to change seed everytime, set it to False
		"fixSeed": True, 
		"randomSeed": 1,

		# settings for grid search
		"lambda":	0,
		"maxTime":	20,
		# ours:0, ramdom:1, epsilon-greedy:2
		"algorithm": 0,
		"epsilon": 0.1,

		# predefined
		"moveCost": moveCost,
		"transitionProb": transitionProb,
		"world": data,
		"startPosition": (data.shape[0]-1, 0)
	}

	if hyperDict["fixSeed"] != True:
		hyperDict["randomSeed"] = time.time()

	return hyperDict


def main():
	# read inputs
	try:
		data, moveCost, transitionProb = readInput()
	except Exception as e:
		return False
	
	# make up hypers
	hyperDict = hypers(data, moveCost, transitionProb)

	# set random sequence
	random.seed(hyperDict["randomSeed"])

	# run the program
	results = search(**hyperDict)

	# generate output
	status = writeFile(results)

	return True


if __name__ == '__main__':
	# set up default display mode
	np.set_printoptions(threshold=sys.maxsize)
	# now ready to go
	main()
# sample cmd line to evoke:
# python3 expSearch.py sample_grid.csv -0.04 0.6