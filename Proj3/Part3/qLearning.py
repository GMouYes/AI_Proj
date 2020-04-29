'''
the actual q learning file
for the exact search algorithm
'''

# import numpy as np

from expSearch import *
import time
import numpy as np
import random

class truck(object):
    """docstring for truck"""
    def __init__(self, capacity, multiplier):
        super(truck, self).__init__()
        self.capacity = capacity
        self.emptySpace = capacity
        self.multiplier = multiplier
        self.currentPos = 0
        self.longestDestination = 0
        self.packageList = []
        self.reward = 0

        # 0: stay; 1: move forward; -1: move backward
        self.nextStep = 0

    def loadPackage(self, packageList):
        # upload the packages to truck until capacity
        # return the remaining packages
        self.packageList += packageList[:self.emptySpace]
        packageList = packageList[self.emptySpace:]
        self.emptySpace = self.capacity - len(self.packageList)
        return packageList

    def decideAction(self, **args):
        # whether to wait:False or start:True
        # TODO: implement
        flag = True
        return flag

    def startDeliver(self, penalty):
        self.nextStep = 1
        self.reward +=  penalty
        return True

    def deliver(self, lengthOfRoad):
        # split packages according to destination
        arrivedPackage = [package for package in self.packageList if package.deliverHouse == self.currentPos]
        self.packageList = [package for package in self.packageList if package.deliverHouse != self.currentPos]
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
        return true

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

    def _getStatus(self):
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
        
        self.warehouse = warehouse(initProb, probUpperBound, probLowerBound, increaseProb, decreaseProb)
        self.truck = truck(capacity, multiplier)

        self.reward = 0
        self.clock = 0
        self.lengthOfRoad = lengthOfRoad
        self.maxTime = maxTime

        self.packageNotOnTruck =  []

    def iteration(self):
        # first check ending standard
        if self.clock >= maxTime:
            return False
        self.clock += 1

        # then update warehouse and truck
        result1 = self.warehouse.updateWarehouse()
        result2 = self.truck.updatePos()

        # load into lists
        if result is not None:
            self.packageNotOnTruck.append(result)

        # is the truck currently at warehouse?
        if self.truck.currentPos == 0:
            # load on truck, update remaining
            self.packageNotOnTruck = self.truck.loadPackage(self.packageNotOnTruck)
            # start the truck or not
            flag = self.truck.decideAction() # you decide the inputs
            if flag:
                self.truck.startDeliver() # calculate reward, leaving warehouse
            else:
                self.truck.wait() # stay for the next clock tick
        # on its way, deliver things
        else:
            self.truck.deliver()
            
        # for every package not yet delivered, calculate penalty
        if len(self.packageNotOnTruck) > 0:
            timeDiff1 = sum([self.clock - package.createTime for package in self.packageNotOnTruck])
        else:
            timeDiff1 = 0

        if len(self.truck.packageList) > 0:
            timeDiff2 = sum([self.clock - package.createTime for package in self.truck.packageList])
        else:
            timeDiff2 = 0

        self.truck.reward -= (timeDiff1 + timeDiff2)
        return True

    def simulation(self):
        while self.iteration():
            True # do nothing
        return # up to you about what to return


def search(**args):
    '''
    placeholder func
    '''
    game = environment()

    return result