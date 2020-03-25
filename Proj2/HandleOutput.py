import numpy

def handleTime(elapsedTime):
    # better format the elapsedTime into hour,min,sec,msec
    timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
                                                        int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
    return timeString


def writeFile(result: dict):
    '''
    result = {
        'clusterCenters': (0,0), # expect a list of tuples of mean and var
        'logLikelihood': -1., # expect a floating number
    }
    '''

    print("OK everything is done now!")
    print("-------------------------------------------")

    print("Best fitting cluster centers:")
    for pair in result["clusterCenters"]:
        print("Mean: {}, Var: {}".format(pair[0], pair[1]))
    print("-------------------------------------------")
    print("Log-likelihood: {:.4f}".format(result["logLikelihood"]))

    return True
