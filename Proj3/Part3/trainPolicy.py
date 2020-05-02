

from expSearch import *

from itertools import product

def trainFunc():
	hypers = {

		# settings for search
		"initCreateProb": [0.13],
		"increaseProb": [0.02],
		"decreaseProb": [-0.02],
		"probUpperBound": [0.25],
		"probLowerBound": [0.05],
		"deliveryMultiplier": [30],
		"defaultMaxTime": [1000],
		
		# not sure if we will use them
		# ours:0, ramdom:1, epsilon-greedy:2
		"lambda":	[0],
		"algorithm": [0],
		"epsilon": [0.1],

		# predefined by cmd line input
		"truckCapacity":	[i for i in range(5,51)],
		"startTruckPenalty":	[-i for i in range(5,101)],
		"lengthOfRoad":	[i for i in range(5,51)],
		"maxTime":	[100]
	}

	hyper = [dict(zip(hypers.keys(),v)) for v in product(*hypers.values())]
	
	startTime = time.time()
	# myCounter = 1
	features,labels = [],[]
	for hyperDict in hyper:
		game = environment(**hyperDict)
		game.simulation()
		features.append(np.array(game.get_features_from_log()))
		labels.append(np.array())

		# print(myCounter)
		myCounter += 1
	endTime = time.time()
	print(endTime-startTime)
	
	features = np.concatenate(features, axis=0)
	labels = np.concatenate(labels, axis=0)

	model = 

def ():
	pass

trainFunc() # takes about 60 secs, if we write logs, then about 90 secs