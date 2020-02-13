import csv
import numpy as np

def printBoard(boardState):
	print(boardState.state)
	return True

def printMoves(moves):
	# sth to implement here
	pass

def readBoard(fileName):
	validList = [str(i) for i in range(10)]
	boardState = []

	with open(fileName, "r") as f:
		csvReader = csv.reader(f)
		for row in csvReader:
			boardState.append([int(item)**2 if item in validList else 0 for item in row])
	returnBoard = board(np.array(boardState))
	return returnBoard

class board(object):
	"""docstring for board"""
	def __init__(self, state=None):
		super(board, self).__init__()
		# state is a 2d ndarray
		# 0 means empty
		# non zeros indicate weights^2 of queens
		self.state = state
		self.h = state.shape[0]
		self.w = state.shape[1] # this is also #Queens

	def cost(self):
		pass

	# @staticmethod
	# def cost(b1, b2):
	# 	# return cost

	def heuristic(self):
		# return heuristic
		return None


	def get_neighbors(self):
		# return all next states and  its cost;
		neighbors = []
		positions = []
		n = len(self.state)
		for i in range(n):
			for j in range(n):
				if self.state[i][j] != ',':
					positions.append((j, self.state[i][j]))
		for i in range(n):
			for j in range(n):
				if self.state[i][j] == ',':
					new_list = self.state[:]
					new_list[i][j], new_list[i][positions[i][0]] = new_list[i][positions[i][0]], new_list[i][j]
					cost = abs(j - positions[i][0])*positions[i][1]**2
					neighbors.append((board(new_list), cost))
		return neighbors



	def finished(self):
		# return boolean
		return self.heuristic() == 0