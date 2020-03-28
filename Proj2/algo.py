import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal
import copy
import matplotlib.pyplot as plt


def Expectation(data, mean, cov, weight):
    # likelihood for each cluster of all data instances
    likelihood = [multivariate_normal.pdf(data, *item) for item in zip(mean, cov)]  # (#clusters, #data)
    # multiply by their weights
    weighted_likelihood = np.array([w * ll for (w, ll) in zip(weight.T, likelihood)])  # (#clusters, #data)
    # avg to guarantee a responsibility/probability sum of 1
    responsibility = weighted_likelihood / np.sum(weighted_likelihood, axis=0)  # (#clusters, #data)

    return responsibility


def Maximization(responsibility, data):
    num_data, dim_data = data.shape
    num_cluster = responsibility.shape[0]

    norm_res = responsibility / np.sum(responsibility, axis=1).reshape(-1, 1)  # (#clusters, #data)

    new_mean = norm_res.dot(data)  # clusters, dim_data

    new_cov = np.zeros((num_cluster, dim_data, dim_data))  # clusters, dim_data, dim_data
    for j in range(num_cluster):
        for n in range(num_data):
            norm_data = data[n, :] - new_mean[j, :]
            new_cov[j] += norm_res[j][n] * norm_data.reshape(-1, 1).dot(norm_data.reshape(1, -1))

    new_weight = np.mean(responsibility, axis=1)

    return new_mean, new_cov, new_weight


def InitializeCluster(data, clusters, random_ratio):
    # the way we initialize:
    # pick instances randomly and calculate parameters
    # we could have run knn first but that takes time
    num_data, dim_data = data.shape
    num_rows = int(num_data * random_ratio)

    mean, cov = [],[]
    for _ in range(clusters):
        indices = np.random.choice(num_data, num_rows)
        sample = data[indices, :]
        mean.append(np.mean(sample, axis=0))
        cov.append(np.cov(sample.T, bias=True))

    mean, cov = np.array(mean), np.array(cov)
    # weights should sum to 1
    weight = np.random.dirichlet(np.ones(clusters), size=1)

    return mean, cov, weight


def Loglikelihood(data, mean, cov, weight):
    likelihood = [multivariate_normal.pdf(data, *item) for item in zip(mean, cov)]
    log_sum = np.sum(np.log(np.sum(weight * np.array(likelihood).T, axis=1)))

    return log_sum

def timeDiff(startTime, period):
    return (time.time() - startTime) < period

def convergence(value1, value2, threshold):
    return abs(value1 - value2) < diff

def updateResult(mean, cov, weight, log_sum, restart, start_time, clusters, num_data):
    # notice: this is the "time" to find the best solution, not total time
    # if you wish to include total time, please add another dict key
    # the same thing applies on entrance "restart"
    result = {
        "clusterCenters": list(zip(mean,cov)),
        "logLikelihood": log_sum,
        "weight": weight,
        "restart": restart,
        "time": time.time() - start_time, 
        "clusters": clusters,
        "BIC": -2 * log_sum + np.log(num_data) * clusters * 2,
    }
    return result

def Clustering(data, clusters, maxTime, random_start_times, random_ratio, start_time, diff):
    num_data, dim_data = data.shape

    for epoch in range(random_start_times):

        mean, cov, weight = InitializeCluster(data, clusters, random_ratio)
        log_sum = Loglikelihood(data, mean, cov, weight)
        result = updateResult(mean, cov, weight, log_sum, epoch+1, \
            start_time, clusters, num_data)

        old_log_sum = log_sum
        while True:
            # expectation
            responsibility = Expectation(data, mean, cov, weight)
            # maximization
            mean, cov, weight = Maximization(responsibility, data)
            # evaluation
            log_sum = Loglikelihood(data, mean, cov, weight)
            # store better global result
            if log_sum > result["logLikelihood"]:
                result = updateResult(mean, cov, weight, log_sum, epoch+1, \
                    start_time, clusters, num_data)
            # begin next random start if the current converged
            if convergence(old_log_sum, log_sum, diff):
                break
            # return global best if time limit is reached
            if not timeDiff(start_time, maxTime):
                return result
            # update old log sum 
            old_log_sum = log_sum

    return result


def EMClustering(data, clusters):

    # change these constants for more tests
    random_start_times = 3
    random_ratio = .3
    maxPeriod = 10 # algo should run no longer than 10 secs
    diff = 0.0001 # convergence threshold

    if clusters > 0:
        start_time = time.time()
        result = Clustering(data, clusters, maxPeriod, random_start_times, random_ratio, start_time, diff)
        return result

    else:
        start_time = time.time()
        period = maxPeriod
        temp_best = float("inf")
        for candidate_cluster in range(1, data.shape[0]):
            temp_result = Clustering(data, candidate_cluster, period, random_start_times, random_ratio, start_time, diff)
            period -= result["time"]
            # store better scores
            if temp_result["BIC"] < temp_best:
                result = copy.deepcopy(temp_result)
                temp_best = temp_result["BIC"]
            # return global best if time limit is reached
            if not timeDiff(start_time, period):
                return result

        return result
    