'''
the actual q learning file
for the exact search algorithm
'''

import time
import numpy as np
import random

from expSearch import *


class truck(object):
    """docstring for truck"""

    def __init__(self, capacity, multiplier, startPenalty):
        super(truck, self).__init__()
        self.capacity = capacity
        self.multiplier = multiplier
        self.currentPos = 0
        self.longestDestination = 0
        self.packageList = []
        self.reward = 0
        self.startPenalty = startPenalty

        # 0: stay; 1: move forward; -1: move backward
        self.nextStep = 0

    def log(self):
        return [self.currentPos, self.reward, self.packageList, self.nextStep]

    def loadPackage(self, packageList):
        # upload the packages to truck until capacity
        # return the remaining packages
        # FIFO
        emptySpace = self.capacity - len(self.packageList)
        self.packageList += packageList[:emptySpace]
        packageList = packageList[emptySpace:]
        return packageList

    def decideAction(self, strategy):
        # whether to wait:False or start:True
        # TODO: implement
        if strategy == 0:
            # dummy strategy 0: always start so long as its not empty
            # this is for debugging
            # comment out for real case
            flag = True if len(self.packageList) > 0 else False

        if strategy == 1:
            # dummy strategy 1: only start when it is full
            # this is for debugging
            # comment out for real case
            flag = True if len(self.packageList) == self.capacity else False

        # you design other strategies

        return flag

    def startDeliver(self):
        self.nextStep = 1
        self.reward += self.startPenalty
        return True

    def deliver(self, lengthOfRoad):
        # split packages according to destination
        arrivedPackage = [item for item in self.packageList if item.deliverHouse == self.currentPos]
        self.packageList = [item for item in self.packageList if item.deliverHouse != self.currentPos]
        # delivery reward
        self.reward += self.multiplier * lengthOfRoad * len(arrivedPackage)
        # update next step strategy
        if len(self.packageList) > 0:
            self.nextStep = 1
        else:
            self.nextStep = -1

        return True

    def wait(self):
        # next step still same place
        self.nextStep = 0
        return True

    def updatePos(self):
        self.currentPos += self.nextStep
        return True


class warehouse(object):
    """docstring for warehouse"""

    def __init__(self, initProb, probUpperBound, probLowerBound, increaseProb, decreaseProb):
        super(warehouse, self).__init__()
        self.prevProb = initProb
        # True for generated, False for not generated
        self.prevStatus = True

        self.probUpperBound = probUpperBound
        self.probLowerBound = probLowerBound
        self.increaseProb = increaseProb
        self.decreaseProb = decreaseProb

    def log(self):
        return [self.prevStatus, self.prevProb]

    def _getProb(self):
        if self.prevStatus:
            prob = min(self.prevProb + self.increaseProb, self.probUpperBound)
        else:
            prob = max(self.prevProb + self.decreaseProb, self.probLowerBound)
        return prob

    def _updateProb(self, prob):
        self.prevProb = prob

    def _updateStatus(self, status):
        self.prevStatus = status

    def _getStatus(self, prob):
        # tell whether it should create new package
        state = random.random()
        if state < prob:
            status = True
        else:
            status = False
        return status

    def _createPackage(self, timestamp, lengthOfRoad):
        deliverHouse = random.randint(1, lengthOfRoad)
        newPackage = package(timestamp, deliverHouse)
        return newPackage

    def updateWarehouse(self, timestamp, lengthOfRoad):
        # get new prob and status
        prob = self._getProb()
        status = self._getStatus(prob)
        # update new prob and status
        self._updateProb(prob)
        self._updateStatus(status)
        # generate new package or not, return object
        if status:
            return self._createPackage(timestamp, lengthOfRoad)
        return None


class package(object):
    """docstring for package"""

    def __init__(self, createTime, deliverHouse):
        super(package, self).__init__()
        self.createTime = createTime
        self.deliverHouse = deliverHouse


class environment(object):
    """docstring for environment"""

    def __init__(self, **args):
        super(environment, self).__init__()

        warehouseDict = {
            "initProb": args["initCreateProb"],
            "probUpperBound": args["probUpperBound"],
            "probLowerBound": args["probLowerBound"],
            "increaseProb": args["increaseProb"],
            "decreaseProb": args["decreaseProb"],
        }

        truckDict = {
            "capacity": args["truckCapacity"],
            "multiplier": args["deliveryMultiplier"],
            "startPenalty": args["startTruckPenalty"]
        }

        self.warehouse = warehouse(**warehouseDict)
        self.truck = truck(**truckDict)

        self.reward = 0
        self.clock = 0
        self.lengthOfRoad = args["lengthOfRoad"]
        self.maxTime = args["maxTime"]

        self.packageNotOnTruck = []

        self.logs = []

    def log(self):
        return [self.truck.log(), self.warehouse.log(), self.packageNotOnTruck]

    def get_features_from_log(self):
        '''
        features to include:
            capacity
            startPenalty
            lengthOfRoad

            #packages on truck
            #packages not on truck

            #prevStatus
            #prevProb
            #longestDis
        '''
        # return all the start points
        logs = self.logs
        features = []
        # print(len(logs))
        # the first start point should be included
        for j in range(len(logs)):
            if logs[j][0][0] == 1 and logs[j][0][3] == 1:
                maxDistance = 0
                if len(logs[j-1][0][2]) > 0:
                    maxDistance = max([item.deliverHouse for item in logs[j-1][0][2]])
                features.append([self.truck.capacity, self.truck.startPenalty, self.lengthOfRoad,
                           len(logs[j-1][0][2]), len(logs[j-1][2]), 
                           logs[j-1][1][0], logs[j-1][1][1], maxDistance])
                break
        

        for i in range(len(logs)-1):
            # prevPos = 1, direction = -1, then next step returned
            if logs[i][0][0] == 1 and logs[i][0][3] == -1:
                maxDistance = 0
                if len(logs[i+1][0][2]) > 0:
                    maxDistance = max([item.deliverHouse for item in logs[i+1][0][2]])
                feature = [self.truck.capacity, self.truck.startPenalty, self.lengthOfRoad,
                           len(logs[i+1][0][2]), len(logs[i+1][2]), 
                           logs[i+1][1][0], logs[i+1][1][1], maxDistance
                           ]
                features.append(feature)
        if len(features) <= 1:
            return []
        features.pop()
        return features

    def get_rewards_from_log(self):
        # return all the start points
        logs = self.logs
        rewards = []
        # the first start point should be included
        for j in range(len(logs)):
            if logs[j][0][0] == 1 and logs[j][0][3] == 1:
                rewards = [logs[j-1][0][1]]
                break


        for i in range(j, len(logs)-1):
            # prevPos = 1, direction = -1, then next step returned
            if logs[i][0][0] == 1 and logs[i][0][3] == -1:
                rewards.append(logs[i+1][0][1])

        if len(rewards) <= 2:
            return []

        period = [rewards[i+1] - rewards[i] for i in range(len(rewards)-1)]

        return period

    def _iteration(self, strategy):

        # first check ending standard
        if self.clock >= self.maxTime:
            return False
        self.clock += 1

        # then update warehouse and truck
        result = self.warehouse.updateWarehouse(self.clock, self.lengthOfRoad)
        _ = self.truck.updatePos()

        self.logs.append(self.log())

        # load into lists
        if result is not None:
            self.packageNotOnTruck.append(result)

        # is the truck currently at warehouse?
        if self.truck.currentPos == 0:
            # load on truck, update remaining
            self.packageNotOnTruck = self.truck.loadPackage(self.packageNotOnTruck)
            # start the truck or not
            flag = self.truck.decideAction(strategy=strategy)  # you decide the inputs
            if flag:
                self.truck.startDeliver()  # calculate reward, leaving warehouse
            else:
                self.truck.wait()  # stay for the next clock tick
        # on its way, deliver things
        else:
            self.truck.deliver(self.lengthOfRoad)

        # for every package not yet delivered, calculate penalty
        if len(self.packageNotOnTruck) > 0:
            timeDiff1 = sum([self.clock - item.createTime for item in self.packageNotOnTruck])
        else:
            timeDiff1 = 0

        if len(self.truck.packageList) > 0:
            timeDiff2 = sum([self.clock - item.createTime for item in self.truck.packageList])
        else:
            timeDiff2 = 0

        self.truck.reward -= (timeDiff1 + timeDiff2)
        return True

    def simulation(self):
        # 1. test: strategy 0: pass!
        # 2. test: strategy 1: pass!
        strategy = 0
        while self._iteration(strategy):
            # do anything

            # prints for debug, comment out in real case 
            # self._stepCheck()

            # 3. test: only do one step, uncomment next line
            # break
            continue

        return True

    def _stepCheck(self):
        print("Clock time:\t", self.clock)
        print("-------------------------------------------")
        print("Warehouse Check:")
        print("Prev Status Create Package:\t", self.warehouse.prevStatus)
        print("Prev Prob Create Package:\t", self.warehouse.prevProb)
        print("#Packages left in house:\t", len(self.packageNotOnTruck))
        print("Destinations:\t\t\t", [item.deliverHouse for item in self.packageNotOnTruck])
        print("-------------------------------------------")
        print("Truck Check:")
        print("Truck Position:\t\t\t", self.truck.currentPos)
        print("#Packages left on truck:\t", len(self.truck.packageList))
        print("Destinations:\t\t\t", [item.deliverHouse for item in self.truck.packageList])
        print("Current Truck Reward:\t\t", self.truck.reward)
        print("")
        return True


def search(**args):
    '''
    '''
    mode = "train"  # test/train
    policy = "function"  # table/function

    if mode == "train":
        if policy == "function":
            model = trainFunc()
        else:
            pass
    else:  # test mode should also separate #TODO
        game = environment(**args)
        game.simulation()

    return {}  # up to designers
