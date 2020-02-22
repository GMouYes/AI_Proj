from HandleInput import *
from HandleOutput import *
from algo import *
from Rules import *
from InstanceTemplate import *

import numpy as np

def search(targetMap,searchType):
    if searchType == "ga":
        result = genetic(targetMap)
    else:
        pass
        #result = HillClimbing(targetMap)
    return result


def main():
    # industrial, commercial, residential, siteMap, searchType = readInput()
    # targetMap = Map(mapState=siteMap, maxIndustrial=industrial, maxCommercial=commercial, maxResidential=residential)
    # result = search(targetMap, searchType)
    # output not done yet
    industrial, commercial, residential, siteMap = readFile("urban1.txt")
    targetMap = Map(mapState=siteMap, maxIndustrial=industrial, maxCommercial=commercial, maxResidential=residential)
    for i in range(50):
        best, zones, runtime = genetic(targetMap, 100, 2, 2, 90)
        print(runtime)
        print(best)
        print([zone.name for zone in zones], [zone.location for zone in zones])
        if best < 12:
            print("not global optimal")
            break
    # return True



if __name__ == '__main__':
    main()