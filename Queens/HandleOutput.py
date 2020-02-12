from Board import *

def handleTime(elapsedTime):
	# better formate the elapsedTime into hour,min,sec,msec
	# sth to implement here
	timeString = ""
	return timeString

def generateOutput(searchResults):
	print("Search Done! Generating output report ...")
	print("")

	print("The initial board state:")
	printBoard(searchResults["initBoard"])
	print("")

	print("#Nodes expanded:", searchResults["expandNodeCount"])
	print("Time elapsed:", handleTime(searchResults["elapsedTime"])
	print("Effective branching factor:", searchResults["branchingFactor"])
	print("Cost of moves:", searchResults["cost"])
	print("Seq of moves:")
	printMoves(searchResults["sequence"])
	print("")

	print("That is all, ty for testing on our program.")
	return True
