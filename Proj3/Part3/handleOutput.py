import pandas as pd


def handleTime(elapsedTime):
    ''' better format the elapsedTime into hour,min,sec,msec '''
    timeString = "{} hour {} min {} sec {} msec".format(elapsedTime // 3600, (elapsedTime % 3600) // 60,
                                                        int(elapsedTime % 60), int(1000 * (elapsedTime % 1)))
    return timeString


def writeFile(result: dict):
    ''' write the output in a cleaner format '''
    with open("Part3_Output.txt", "w+") as f:
        f.write("OK everything is done now!\n")
        f.write("-------------------------------------------\n")

        f.write("Training time: {}\n".format(handleTime(result["time"])))
        f.write("Best Policy:\n")
        pd.set_option('display.max_rows', None)
        f.write(str(result["policy"]) + '\n')
        f.write("-------------------------------------------\n")
        f.write("Average Reward:\n")
        f.write(str(result["reward"]) + '\n')

    with open("Part3_Output.txt", "r") as f:
        for line in f:
            print(line)

    return True
