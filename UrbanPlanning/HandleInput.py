def readFile(fileName):
    with open(fileName, "r") as f:
        info = [line.strip("\n") for line in f]

    industrial, commercial, residential = info[:3]
    siteMap = [list(row.split(",")) for row in info[3:]]

    return industrial, commercial, residential, siteMap
