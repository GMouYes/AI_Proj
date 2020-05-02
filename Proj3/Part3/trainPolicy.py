

from expSearch import *
from model import *

from itertools import product
from statistics import median

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

		feature = game.get_features_from_log()
		label = game.get_rewards_from_log()

		if len(feature) > 0 and len(label) > 0:
			# checker for safety
			# if len(feature) != len(label):
				# print(len(feature), len(label))
				# break
			features.append(np.array(feature))
			labels.append(np.array(label))

		# print(myCounter)
		# myCounter += 1
	endTime = time.time()
	print(endTime-startTime)
	
	features = np.concatenate(features, axis=0)
	labels = np.concatenate(labels, axis=0)

	labels = labels > median(labels)

	print("shape:", features.shape, labels.shape)
	startTime = time.time()
	model = buildModel(features,labels)
	endTime = time.time()
	print(endTime-startTime)
	return model

trainFunc() # takes about 60 secs, if we write logs, then about 150 secs