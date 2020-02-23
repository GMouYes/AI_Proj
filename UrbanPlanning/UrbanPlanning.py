from HandleInput import *
from HandleOutput import *
from algo import *
from Rules import *

def search(targetMap,searchType):
	if searchType == "ga":
		result = genetic(targetMap)
	elif searchType == "hc":
		result = greedyHillClimb(targetMap)
	else:
		result = greedyHillClimb(targetMap, mode="super_greedy")
	#print(result)
	return result

def main():
	try:
		industrial, commercial, residential, siteMap, searchType = readInput()
	except Exception as e:
		return False

	targetMap = Map(mapState=siteMap, maxIndustrial=industrial, maxCommercial=commercial, maxResidential=residential)
	result = search(targetMap, searchType)
	
	writeFile(result)
	return True


if __name__ == '__main__':
	main()
