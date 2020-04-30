import numpy as np


def handleTime(elapsedTime):
    ''' better format the elapsedTime into hour,min,sec,msec '''
    timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
                                                        int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
    return timeString


def translatePolicy(policy):
    height,width = policy.shape
    directionMapping = {
        0:  "up",
        1:  "right",
        2:  "down",
        3:  "left",
    }
    newPolicy = np.zeros((height,width), dtype=str)

    for i in range(height):
        for j in range(width):
            newPolicy[i][j] = directionMapping[policy[i][j]]
    return newPolicy


def writeFile(result: dict):
    ''' write the output in a cleaner format '''
    print("OK everything is done now!")
    print("-------------------------------------------")

    print("Training time:", result["time"])
    print("Best Policy:")
    print("2D matrix of shape Height*Width")
    print(translatePolicy(result["policy"]))
    print("-------------------------------------------")
    print("Q values:")
    print("\nUp:")
    print(result["Q"][:, :, 0])
    print("\nRight:")
    print(result["Q"][:, :, 1])
    print("\nDown:")
    print(result["Q"][:, :, 2])
    print("\nLeft:")
    print(result["Q"][:, :, 3])
    print("-------------------------------------------")
    print("Expected reward:", result["reward"])

    return True
