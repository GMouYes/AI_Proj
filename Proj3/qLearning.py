'''
the actual q learning file
for the exact search algorithm
'''

#import numpy as np

from expSearch import *

def search(**args):
	'''
	placeholder function
	# TODO: implement details
	'''
	height, width = args["world"].shape

	# expected return: dict of following
	policy = np.zeros((height,width)) # 0,1,2,3 indicating up, right, down, left
	reward = np.zeros((height,width)) # real values
	
	result = {
		"policy": policy,
		"reward": reward,
		# others to be determined
		# time might be important
	}

	return result

def lookForPath():
	stepCounter = 0
	while True:
		if isEnd():
			reward = pathReward(position, stepCounter)
			break
		stepCounter += 1
		direction = policyDirection()
		direction = actualDirection(direction, ratio)
		position = actualPosition(world, position, direction)
	return

def pathReward():
	return

def isEnd(world, position: tuple):
	if world[position] != 0:
		return False
	return True

def policyDirection():
	# return direction based on algorithm
	# still placeholder
	return 0

def actualDirection(direction, ratio):
	guess = random.random()
	if guess < ratio:
		return direction
	if guess < (1.+ratio)/2.:
		return (direction-1)%4
	return (direction+1)/4

def actualPosition(world, position, direction):
	y,x = position

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

