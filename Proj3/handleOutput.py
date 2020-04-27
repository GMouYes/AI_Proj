def handleTime(elapsedTime):
    ''' better format the elapsedTime into hour,min,sec,msec '''
    timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
                                                        int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
    return timeString


def writeFile(result: dict):
    ''' write the output in a cleaner format '''
    print("OK everything is done now!")
    print("-------------------------------------------")

    print("Training time:", result["time"])
    print("Best Policy:")
    print(result["policy"])
    print("-------------------------------------------")
    print("Best Reward:")
    print(result["reward"])

    return True