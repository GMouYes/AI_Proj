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
    def __init__(self, capacity):
        super(truck, self).__init__()
        self.capacity = capacity
        self.currentPos = 0
        self.longestDestination = 0

        # "wait", "delivery", "returning"
        self.status = "wait"

    def deliver(self):
        pass

    def wait(self):
        pass

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

    def getProb(self):
        if self.prevStatus:
            prob = min(self.prevProb + self.increaseProb, self.probUpperBound)
        else:
            prob = max(self.prevProb + self.decreaseProb, self.probLowerBound)
        return prob

    def updateProb(self, prob):
        self.prevProb = prob

    def updateStatus(self, status):
        self.prevStatus = status

    def getStatus(self):
        prob = self.getProb()
        state = random.random()
        if state < prob:
            status = True
        else:
            status = False
        return status
        
    def createPackage(self, timestamp, lengthOfRoad):
        deliverHouse = random.randint(1, lengthOfRoad)
        newPackage = package(timestamp, deliverHouse)
        return newPackage

        
class package(object):
    """docstring for package"""
    def __init__(self, createTime, deliverHouse):
        super(package, self).__init__()
        self.createTime = createTime
        self.deliverHouse = deliverHouse
        
        

def search(**args):
    '''
    placeholder func
    '''

    return result