import csv

def printBoard(boardState):
	# sth to implement here
	pass

def printMoves(moves):
	# sth to implement here
	pass

def readBoard(fileName):
	# sth to implement here
	returnBoard = board()
	return returnBoard

class board(object):
	"""docstring for board"""
	def __init__(self, state: list):
		# sth to implement here
		super(board, self).__init__()
		self.state = state

	# @staticmethod
	# def cost(b1, b2):
	# 	# return cost

	def heuristic(self):
		# return heuristic


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