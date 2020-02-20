from HandleInput import *
from HandleOutput import *
from algo import *
from Rules import *
from InstanceTemplate import *

import numpy as np

def search(targetMap,searchType):
	if searchType == "ga":
		result = Genetic(targetMap)
	else:
		#result = HillClimbing(targetMap)
	return Result

def main():
	industrial, commercial, residential, siteMap, searchType = readInput()
	targetMap = Map(mapState=siteMap, maxIndustrial=industrial, maxCommercial=commercial, maxResidential=residential)
	result = search(targetMap,searchType)
	# output not done yet

	return True

if __name__ == '__main__':
	main()