import Rules
import numpy as np


def createSiteX(location: tuple):
    toxicScore = {
        "I": {"distance": 2, "score": -10},
        "C": {"distance": 2, "score": -20},
        "R": {"distance": 2, "score": -20},
    }

    build_on_cost = -np.inf  # a huge number forcing not to build on
    newSite = Rules.Site(name="X", score=toxicScore, build_on_cost=build_on_cost, location=location)

    return newSite


def createSiteS(location: tuple):
    sceneryScore = {
        "I": {"distance": -1, "score": 0},
        "C": {"distance": -1, "score": 0},
        "R": {"distance": 2, "score": 10},
    }

    build_on_cost = 1
    newSite = Rules.Site(name="S", score=sceneryScore, build_on_cost=build_on_cost, location=location)

    return newSite


def createSite(siteName: str, location: tuple):
    return createSiteX(location) if siteName == "X" else createSiteS(location)


def createZoneI(location):
    score = {
        "I": {"distance": 2, "score": 2},
        "C": {"distance": -1, "score": 0},
        "R": {"distance": -1, "score": 0},
    }
    newZone = Rules.Zone(name="I", score=score, location=location)
    return newZone


def createZoneC(location):
    score = {
        "I": {"distance": -1, "score": 0},
        "C": {"distance": 2, "score": -4},
        "R": {"distance": 3, "score": 4},
    }
    newZone = Rules.Zone(name="C", score=score, location=location)
    return newZone


def createZoneR(location):
    score = {
        "I": {"distance": 3, "score": -5},
        "C": {"distance": 3, "score": 4},
        "R": {"distance": -1, "score": 0},
    }
    newZone = Rules.Zone(name="R", score=score, location=location)
    return newZone


def createZone(zoneName, location: tuple):
    if zoneName == "I":
        return createZoneI(location)
    elif zoneName == "C":
        return createZoneC(location)
    else:
        return createZoneR(location)
