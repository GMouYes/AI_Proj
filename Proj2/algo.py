import numpy as np
import time
import random
import math

def Gaussian(x, mu=0.0, sigma=1.0):
    x = (x - mu) / sigma
    return np.exp(-x**2/2.0) / math.sqrt(2.0*math.pi) / sigma

# placeholder function
def Expectation():
    pass

# placeholder function
def Maximization():
    pass

def InitialSampling():
	pass

def Convergence():
	pass


# placeholder function
def EMClustering(data, clusters):
	num_data,dim_data = data.shape

	# change these constants for more tests
	random_start_times = 10
	sample_magnitude = 20
    for _ in range(random_start_times):
    	InitialSampling()
    	while not Convergence():
	        Expectation() # you define your own input/output
	        Maximization() # you define your own input/output
	        updateBestResult()

    return None # you have to return the required dict
    