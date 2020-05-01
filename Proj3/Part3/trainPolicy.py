

def trainingFunc():
	hyperDict = {

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
		"maxTime":	[1000]
	}

	new_paras = [dict(zip(hyperDict.keys(),v)) for v in itertools.product(*hyperDict.values())]