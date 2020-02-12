import sys
from Board import *

def formatError():
	print("There's some problem with reading inputs, please check again:")
	print("You should have 3 explicit inputs.")
	print("1. Initial board state, indicated by the complete path of a csv file.")
	print("2. Type of searching algo, \"1\" for A star, \"2\" for greedy hill climbing")
	print("3. Heuristic function, \"H1\" or \"H2\", not case sensitive")
	return True

def repeatChoice(initBoard,searchType,heuristicFunc):
	print("Input processing complete, you have chosen:")
	print("")

	print("Initial Board State:")
	printBoard(initBoard)
	#print("")

	print("Searching Algo:", end=" ")
	if searchType == "1":
		print("A Star")
	else:
		print("Greedy Hill Climbing")
	#print("")

	print("Heuristic Function:", heuristicFunc)
	#print("")

	print("Now searching, this may take some time ...")
	#print("")
	return True

def readInut():
	# check passing in parameters
	if len(sys.argv) < 4:
		formatError()
		return False

	boardName = sys.argv[1]
	searchType = sys.argv[2]
	heuristicFunc = sys.argv[3].lower()

	# check legal input strings
	try:
		initBoard = readBoard(boardName)
	except Exception as e:
		formatError()
		return False

	if searchType not in ["1", "2"]:
		formatError()
		return False

	if heuristicFunc not in ["h1", "h2"]:
		formatError()
		return False

	return initBoard,searchType,heuristicFunc


