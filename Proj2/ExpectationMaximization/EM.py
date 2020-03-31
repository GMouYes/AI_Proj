from HandleInput import *
from HandleOutput import *
from algo import *

def main():
	# read data and interpret command
    try:
        data, clusters = readInput()
    except Exception as e:
        return False

    # run experiment and return results
    result = EMClustering(data, clusters)

    # generate formatted output
    writeFile(result)

    # successfully done
    return True

if __name__ == '__main__':
    main()

# sample test cmd line: python3 EM.py sample_EM_data.csv 3