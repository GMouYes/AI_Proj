from HandleInput import *
from HandleOutput import *
from algo import *

def search(data, clusters):
    '''
    result = {
        'clusterCenters': [(0.12345,0.54321), (1.23456, 6.54321)], # expect a list of tuples of mean and var
        'logLikelihood': -1., # expect a floating number
    }
    '''

    # you have to return the required dictionary
    if clusters > 0:
        result = EMClustering(data, clusters)

        # print(result)
        return result
    else:
        best_cluster, BIC, result = EMClustering(data, clusters)
        return best_cluster, BIC, result


def main():
    try:
        data, clusters = readInput()
    except Exception as e:
        return False

    if clusters > 0:
        result = search(data, clusters)

        writeFile(result)
        return True
    else:
        best_cluster, BIC, result = search(data, clusters)
        writeFile_bic(best_cluster, BIC, result)
        return True


if __name__ == '__main__':
    main()

# sample test cmd line: python3 EM.py sample_EM_data.csv 3