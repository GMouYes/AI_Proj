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
		self.height = state.shape[0]
		self.width = state.shape[1] # this is also #Queens
		# position of all queens
		self.queenPos = np.argwhere(state>0)

		# all attack weight^2 lighter list
		self.lightAttackList = self.lighterAttackPieceList()

		# two heuristic values
		self.h1 = min(lightAttackList)
		self.h2 = sum(lightAttackList)

	def ifAttack(self,index1,index2):
		# check if two pieces are attacking each other
		# index: np.ndarray [row,col] of queens

		# this method does not check for same queen (invalid input)
		# this method does not check if there is/are a/two queen(s)

		row1,col1 = index1
		row2,col2 = index2

		# horizontal
		if row1 == row2:
			return True
		# diagonal
		if abs(row1-row2) == abs(col1-col2):
			return True
		return False

	def ifValidQueen(self,row,col):
		# check if the given position has a queen
		return True if self.state[row][col] > 0 else False

	def lighterAttackPieceList(self):
		# returns the list of lighter weight^2 of all attacking pairs
		returnList = []
		for index1 in range(self.queenPos.shape[0]):
			for index2 in range(index1+1,self.queenPos.shape[0]):
				pos1 = self.queenPos[index1]
				pos2 = self.queenPos[index2]
				if ifAttack(pos1,pos2):
					weight1 = self.state[pos1[0]][pos1[1]]
					weight2 = self.state[pos2[0]][pos2[1]]
					returnList.append(min(weight1,weight2))
		return returnList

	def cost(self,row,col,move):
		# row: row index of queen
		# col: col index of queen
		# move: distance to move

		# this method does not check valid moves
		# this method does not check for a queen
		return self.state[row][col] * abs(move)

	def ifValidMove(self,start_index,move):
		# move: - up, + down
		# we can only move queens
		if not ifValidQueen(start_index[0],start_index[1]):
			return False
		# we cannot move out of boundary
		if (start_index[0]+move)<0 or (start_index[1]+move)>self.height-1:
			return False
		return True
		
	def heuristic(self,h_type):
		if h_type == "h1":
			return self.h1
		else:
			return self.h2
	'''
	def get_neighbors(self):
		# return all next states and its cost;
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
	'''