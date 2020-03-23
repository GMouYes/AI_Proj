from HandleInput import *
from HandleOutput import *

def search():
    '''
    result = {
        'clusterCenters': [(0.12345,0.54321), (1.23456, 6.54321)], # expect a list of tuples of mean and var
        'logLikelihood': -1., # expect a floating number
    }
    '''

    # you have to return the required dictionary
    result = EMClustering(data, clusters)

    # print(result)
    return result


def main():
    try:
        data, clusters = readInput()
    except Exception as e:
        return False

    result = search(data, clusters)

    writeFile(result)
    return True


if __name__ == '__main__':
	# set up random seed
	seed = 1
	random.seed(seed)
	np.random.seed(seed)

    main()

# sample test cmd line: python3 EM.py sample_EM_data.csv 3