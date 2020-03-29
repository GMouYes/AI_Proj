import numpy

def handleTime(elapsedTime):
    # better format the elapsedTime into hour,min,sec,msec
    timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
                                                        int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
    return timeString


def writeFile(result: dict):

    print("OK everything is done now!")
    print("-------------------------------------------")

    print("Best fitting clusters:")
    print("Number of clusters: {}".format(result["num_clusters"]))
    for item in result["clusters"]:
        print("Mean:\n{}".format(item[0]))
        print("Cov:\n{}".format(item[1]))
        print("Weight:\n{:.4f}".format(item[2]))
    print("-------------------------------------------")

    print("Total Log-likelihood: {:.4f}".format(result["logLikelihood"]))
    print("Model BIC: {:4f}".format(result["BIC"]))
    print("-------------------------------------------")

    print("Simulation statistics:")
    print("Time: {}".format(handleTime(result["time"])))
    print("#Restarts: {}".format(result["restart"]))

    return True