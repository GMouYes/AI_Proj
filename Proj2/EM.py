from HandleInput import *
from HandleOutput import *
from algo import *

def main():
    try:
        data, clusters = readInput()
    except Exception as e:
        return False

    result = EMClustering(data, clusters)
    writeFile(result)
    return True

if __name__ == '__main__':
    main()

# sample test cmd line: python3 EM.py sample_EM_data.csv 3