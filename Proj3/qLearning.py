'''
the actual q learning file
for the exact search algorithm
'''

#import numpy as np

from expSearch import *
import time

def search(**args):
	'''
	placeholder function
	# TODO: implement details
	'''
	world = args["world"]
	height, width = world.shape
	searchType = args["algorithm"]
	startPosition = args["startPosition"]
	moveCost = args["moveCost"]
	maxTime = args["maxTime"]
	transitionProb = args["transitionProb"]

	# expected return: dict of following
	policy = np.zeros((height,width)) # 0,1,2,3 indicating up, right, down, left
	reward = np.zeros((height,width)) # real values
	
	result = {
		"policy": policy,
		"reward": reward,
		"time": 0,
		# others to be determined
	}

	lookForPath(startPosition, world, moveCost, maxTime, 0, transitionProb)

	return result

def translateProcedure(stepCounter, position, chooseDirection, direction, newPosition):
	directionDict = {
		0:	"up",
		1:	"right",
		2:	"down",
		3:	"left",
	}

	print("Step", stepCounter)
	print("startPosition:", position)
	print("Choosen direction:", directionDict[chooseDirection])
	print("Actual direction:", directionDict[direction])
	print("Arrival position:", newPosition)
	print("")
	return

def endState(position, world, reward):
	print("Reach end pos:", position)
	print("Final reward:", world[position])
	print("Overall reward:", reward)
	print("")

def lookForPath(startPosition, world, moveCost, maxTime, searchType, ratio):
	stepCounter = 0
	position = startPosition
	startTime = time.time()
	while True:
		if isEnd(world, position):
			reward = pathReward(world, position, stepCounter, moveCost)

			endState(position, world, reward)

			break

		if time.time() - startTime > maxTime:
			print("Max time reached, ending at step", stepCounter)
			break

		stepCounter += 1
		chooseDirection = policyDirection(searchType)
		direction = actualDirection(chooseDirection, ratio)
		newPosition = actualPosition(world, position, direction)

		translateProcedure(stepCounter, position, chooseDirection, direction, newPosition)

		position = newPosition
		
	return

def updatePolicy():
	pass

def updateReward():
	pass


def pathReward(world, position, stepCounter, moveCost):
	finalReward = world[position]
	return finalReward + stepCounter * moveCost


def isEnd(world, position: tuple):
	if world[position] != 0.:
		return True
	return False

def policyDirection(searchType):
	# return direction based on algorithm
	# still placeholder
	if searchType == 0:
		return random.randint(0,3)
	return 0

def actualDirection(direction, ratio):
	guess = random.random()
	if guess < ratio:
		return direction

	if guess < (1.+ratio)/2.:
		return (direction-1)%4

	return (direction+1)%4

def actualPosition(world, position, direction):
	y,x = position
	height, width = world.shape

	if direction == 0:
		# up	
		if y == 0:
			return (y,x)
		return (y-1,x)

	if direction == 1:
		# right
		if x == width-1:
			return (y,x)
		return (y,x+1)

	if direction == 2:
		# down
		if y == height-1:
			return (y,x)
		return (y+1, x)

	if direction == 3:
		# left
		if x == 0:
			return (y,x)
		return (y,x-1)

