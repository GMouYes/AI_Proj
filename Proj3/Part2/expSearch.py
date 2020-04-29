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

def hypers(data, moveCost, transitionProb, seed):
	'''
	all hypers should be defined here
	returning the dict of them
	'''

	hyperDict = {
		"randomSeed": seed,

		# settings for grid search
		"lambda":	0,
		"maxTime":	20,
		"tol":		5e-4,
		# ours:0, ramdom:1, epsilon-greedy:2
		"algorithm": 0,
		"epsilon": 0.1,

		# predefined
		"moveCost": moveCost,
		"transitionProb": transitionProb,
		"world": data,
		"startPosition": (data.shape[0]-1, 0)
	}

	return hyperDict


def main(seed=1):
	# read inputs
	try:
		data, moveCost, transitionProb = readInput()
	except Exception as e:
		return False
	
	# make up hypers
	hyperDict = hypers(data, moveCost, transitionProb, seed)

	# run the program
	results = search(**hyperDict)

	# generate output
	status = writeFile(results)

	return True


if __name__ == '__main__':
	# set up default display mode
	np.set_printoptions(threshold=sys.maxsize)
	# set random sequence
	seed = 1
	# seed = time.time()
	random.seed(seed)
	np.random.seed(seed)

	# now ready to go
	main(seed=seed)

# sample cmd line to evoke:
# python3 expSearch.py sample_grid.csv -0.04 0.6