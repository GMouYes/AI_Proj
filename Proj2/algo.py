import numpy as np
import time
import random
import math
from scipy.stats import multivariate_normal

def Expectation(data, mean, cov, weight):
	# likelihood for each cluster of all data instances
    likelihood = [multivariate_normal.pdf(data, *item) for item in zip(mean,cov)] # (#clusters, #data)
    # multiply by their weights
    weighted_likelihood = np.array([w * ll for (w, ll) in zip(weight, likelihood)]) # (#clusters, #data)
    # avg to guarantee a responsibility/probability sum of 1
    responsibility = weighted_likelihood / np.sum(weighted_likelihood, axis=0) # (#clusters, #data)

    return responsibility

def Maximization(responsibility, data):
	num_data,dim_data = data.shape
	num_cluster = responsibility.shape[0]

	norm_res = responsibility / np.sum(responsibility, axis=1).reshape(-1,1) # same shape

	# new_mean = np.zeros((num_cluster, dim_data))
	# for j in range(num_cluster):
	# 	for n in range(num_data):
	# 		new_mean[j] += norm_res[j][n] * data[n,:]

    new_mean = norm_res.dot(data) # clusters, dim_data

    new_cov = np.zeros((num_cluster, dim_data,dim_data)) # clusters, dim_data, dim_data
    for j in range(num_cluster):
		for n in range(num_data):
			norm_data = data[n,:] - new_mean[j,:]
			new_cov[j] += norm_res[j][n] * norm_data.reshape(-1,1).dot(norm_data.reshape(1,-1))

	new_weight = np.mean(responsibility, axis=1)

	return new_mean, new_cov, new_wweight

def InitializeCluster(data, clusters, random_ratio):
	# the way we initialize:
	# pick instances randomly and calculate parameters
	num_data,dim_data = data.shape
	num_rows = int(num_data * random_ratio)

	mean,cov = []
	for _ in range(clusters):
		indices = np.random.choice(num_data, num_rows)
		sample = data[indices,:]
		mean.append(np.mean(sample, axis=0))
		cov.append(np.cov(sample.T, bias=True))

	mean, cov = np.array(mean), np.array(cov)
	# weights should sum to 1
	weight = np.random.dirichlet(np.ones(clusters),size=1)

	return mean, cov, weight

def Convergence(flag):
	# use sum of log-likelihood
	# multivariate_normal.logpdf()
	pass


# placeholder function
def EMClustering(data, clusters):
	num_data,dim_data = data.shape

	# change these constants for more tests
	random_start_times = 10
	random_ratio = .3

    for _ in range(random_start_times):
    	mean, cov, weight = InitializeCluster(data, clusters, random_ratio)
    	while not Convergence(flag): # you define your own input/output
    		# expectation
	        responsibility = Expectation(data, mean, cov, weight) 
	        # maximization
	        mean, cov, weight = Maximization(responsibility, data, mean, cov, weight)
	        # evaluation
	        flag = updateBestResult() # you define your own input/output

    return None # you have to return the required dict
    