import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal
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

    norm_res = responsibility / np.sum(responsibility, axis=1).reshape(-1, 1)  # same shape

    # new_mean = np.zeros((num_cluster, dim_data))
    # for j in range(num_cluster):
    #   for n in range(num_data):
    #       new_mean[j] += norm_res[j][n] * data[n,:]


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
    # use sum of log-likelihood
    # multivariate_normal.logpdf()
    likelihood = [multivariate_normal.pdf(data, *item) for item in zip(mean, cov)]
    weighted_likelihood = np.array([w * ll for (w, ll) in zip(weight.T, likelihood)])
    responsibility = weighted_likelihood / np.sum(weighted_likelihood, axis=0)
    log_sum = np.sum(np.log(np.sum(weight * responsibility.T, axis=1)))

    return log_sum

# placeholder function
def EMClustering(data, clusters):
    num_data, dim_data = data.shape

    # change these constants for more tests
    random_start_times = 0
    random_start_max = 5
    random_ratio = .3

    diff = 0.001
    logLikelihood = [1]
    result ={}
    start_time = time.time()

    if clusters > 0:
        while random_start_times <= random_start_max and time.time() - start_time < 10:
            mean, cov, weight = InitializeCluster(data, clusters, random_ratio)
            temp = 0
            while abs(logLikelihood[-1]- temp) > diff and time.time() - start_time < 10:  # you define your own input/output
                temp = logLikelihood[-1]
                # expectation
                responsibility = Expectation(data, mean, cov, weight)
                # maximization
                mean, cov, weight = Maximization(responsibility, data)
                # evaluation
                log_sum = Loglikelihood(data, mean, cov, weight)

                if log_sum > temp:
                    result['clusterCenters'] = [item for item in zip(mean,cov)]
                    result['logLikelihood'] = logLikelihood[-1]

                logLikelihood.append(log_sum)

            random_start_times += 1
        # print(random_start_times)
        # print(logLikelihood)
        #
        # plt.plot(logLikelihood[1:])
        # plt.xlabel('number of iterations')
        # plt.ylabel('Log-likelihood')
        # plt.show()

        return result

    else:
        clusters = 1

        BIC = {}
        final_result = {}
        while True and time.time() - start_time < 10:
            random_ratio =  .3
            temp = 0
            mean, cov, weight = InitializeCluster(data, clusters, random_ratio)
            result ={}
            logLikelihood = [1]

            while abs(logLikelihood[-1]- temp) > diff and time.time() - start_time < 10:  # you define your own input/output
                temp = logLikelihood[-1]
                # expectation
                responsibility = Expectation(data, mean, cov, weight)
                # maximization
                mean, cov, weight = Maximization(responsibility, data)
                # evaluation
                log_sum = Loglikelihood(data, mean, cov, weight)

                if log_sum > temp:
                    result['clusterCenters'] = [item for item in zip(mean,cov)]
                    result['logLikelihood'] = logLikelihood[-1]
                    BIC[clusters] = -2 * logLikelihood[-1] + np.log(len(data)) * clusters * 2

                logLikelihood.append(log_sum)
            final_result[clusters] = result
            clusters += 1

        best_cluster = max(BIC, key=BIC.get)
        return best_cluster,BIC[best_cluster],final_result[best_cluster]

      # you have to return the required dict
if __name__ == '__main__':
    data = pd.read_csv('sample EM data.csv',header=None).values
    best_cluster, value,final = EMClustering(data, 0)
    print(best_cluster, value,final)

#     result = EMClustering(data, 3)
#     print(result)