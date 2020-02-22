import InstanceTemplate as it
import copy


def get_distance(loc1, loc2):  # manhattan distance
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


def countZone(zoneType, zoneList):
    return sum([zone.name == zoneType for zone in zoneList])


def ifValidZoneList(zoneList, urbanmap):
    # check if it exceed limit
    if countZone("I", zoneList) > urbanmap.maxIndustrial:
        return False
    if countZone("R", zoneList) > urbanmap.maxResidential:
        return False
    if countZone("C", zoneList) > urbanmap.maxCommercial:
        return False
    # if countZone("I", zoneList)+countZone("R", zoneList)+countZone("C", zoneList) == 0:
    #     return False

    # Check if there are areas available for building
    siteList = urbanmap.siteList
    num_sites_built_on = len([site for site in siteList for zone in zoneList if site.location == zone.location])
    if len(zoneList) + len(siteList) - num_sites_built_on > urbanmap.mapState.shape[0] * urbanmap.mapState.shape[1]:
        return False

    # check if two zone at same location
    length = len(zoneList)
    for index1 in range(length):
        for index2 in range(index1 + 1, length):
            if zoneList[index1].location == zoneList[index2].location:
                return False

    return True


def get_neighbors(zone_list, urbanmap):
    # Four possible operations per state: Moving, deleting, swapping, and creating zones
    neighbors = []
    # Move:
    for zone in zone_list:
        for row in range(urbanmap.mapState.shape[0]):
            for col in range(urbanmap.mapState.shape[1]):
                cur_pos = zone.location
                if (row, col) != cur_pos:
                    changed_zone = copy.copy(zone)
                    zone.location = (row, col)
                    if ifValidZoneList(zone_list, urbanmap):
                        move = {
                            "zone_list": zone_list,
                            "operation": "move",
                            "old_zone": changed_zone,
                            "new_zone": copy.copy(zone),
                            "score": urbanmap.get_score(zone_list)
                        }
                        neighbors.append(move)

    # Delete:
    for zone in zone_list:
        zone_list_copy = copy.copy(zone_list)
        zone_list_copy.remove(zone)
        if ifValidZoneList(zone_list_copy, urbanmap):
            move = {
                "zone_list": zone_list_copy,
                "operation": "delete",
                "old_zone": copy.copy(zone),
                "score": urbanmap.get_score(zone_list_copy)
            }
            neighbors.append(move)

    # Swap
    zone_list_copy = copy.copy(zone_list)
    for ind1 in range(len(zone_list)):
        for ind2 in range(ind1 + 1, len(zone_list)):
            if zone_list[ind1].name != zone_list[ind2].name:
                zone_list[ind1].location, zone_list[ind2].location = zone_list[ind2].location, zone_list[ind1].location
                if ifValidZoneList(zone_list, urbanmap):
                    move = {
                        "zone_list": zone_list_copy,
                        "operation": "swap",
                        "new_zones": (copy.copy(zone_list[ind1]), copy.copy(zone_list[ind2])),
                        "score": urbanmap.get_score(zone_list)
                    }
                    zone_list[ind1].location, zone_list[ind2].location = zone_list[ind2].location, zone_list[
                        ind1].location
                    move["old_zones"] = (copy.copy(zone_list[ind1]), copy.copy(zone_list[ind2]))
                    neighbors.append(move)

    # Create
    zone_max_dict = {"C": urbanmap.maxCommercial, "I": urbanmap.maxIndustrial, "R": urbanmap.maxResidential}
    for zone_type in ["C", "I", "R"]:
        if countZone(zone_type, zone_list) < zone_max_dict[zone_type]:
            for row in range(urbanmap.mapState.shape[0]):
                for col in range(urbanmap.mapState.shape[1]):
                    zone_list_copy = copy.copy(zone_list)
                    new_zone = it.createZone(zone_type, (row, col))
                    zone_list_copy.append(new_zone)
                    if ifValidZoneList(zone_list_copy, urbanmap):
                        move = {
                            "zone_list": zone_list_copy,
                            "operation": "create",
                            "new_zone": copy.copy(new_zone),
                            "score": urbanmap.get_score(zone_list_copy)
                        }
                        neighbors.append(move)

    return neighbors


class Site(object):
    """docstring for Site"""

    def __init__(self, name: str, score: dict, build_on_cost: int, location: tuple):
        super(Site, self).__init__()
        self.name = name
        self.score = score
        self.backupScore = copy.deepcopy(score)
        self.build_on_cost = build_on_cost
        self.location = location

    def get_score(self, newZone):
        distance = get_distance(self.location, newZone.location)

        if distance == 0:
            return self.build_on_cost
        elif distance > self.score[newZone.name]["distance"]:
            return 0  # not in range
        else:
            return self.score[newZone.name]["score"]

    def buildOn(self):
        for zoneName in self.score:
            self.score[zoneName]["score"] = 0  # update to destroy
        return True

    def recoverSite(self):
        self.score = copy.deepcopy(self.backupScore)
        return True


class Zone(object):
    """docstring for zone"""

    def __init__(self, name: str, score: dict, location: tuple):
        super(Zone, self).__init__()
        self.name = name
        self.score = score
        self.location = location

    def __lt__(self, other):
        return self.location[0] < other.location[0]

    def get_score(self, newZone):
        distance = get_distance(self.location, newZone.location)

        if distance == 0:
            return 0  # same zone
        elif distance > self.score[newZone.name]["distance"]:
            return 0  # not in range
        else:
            return self.score[newZone.name]["score"]


class Map(object):
    """docstring for Map"""

    def __init__(self, mapState, maxIndustrial, maxCommercial, maxResidential):
        super(Map, self).__init__()
        self.mapState = mapState  # a 2D numpy matrix
        self.siteList = self.get_site_List()
        self.maxIndustrial = maxIndustrial
        self.maxCommercial = maxCommercial
        self.maxResidential = maxResidential

    def get_site_List(self):
        siteList = []
        for row in range(self.mapState.shape[0]):
            for col in range(self.mapState.shape[1]):
                state = self.mapState[row, col]
                if state in ["X", "S"]:
                    siteList.append(it.createSite(siteName=state, location=(row, col)))
        return siteList

    def get_score(self, zoneList):  # this is all you need
        self.checkBuildOn(zoneList)
        score = self.get_zone_zone_score(zoneList) + \
                self.get_zone_site_score(zoneList) - \
                self.get_zone_cost(zoneList)

        for site in self.siteList:
            site.recoverSite()
        return score  # the larger the better!!!

    def get_zone_zone_score(self, zoneList):
        zone_zone_score = 0
        for zone1 in zoneList:
            for zone2 in zoneList:
                zone_zone_score += zone1.get_score(zone2)
        return zone_zone_score

    def get_zone_site_score(self, zoneList):
        zone_site_score = 0
        for site in self.siteList:
            for zone in zoneList:
                zone_site_score += site.get_score(zone)
        return zone_site_score

    def checkBuildOn(self, zoneList):
        for site in self.siteList:
            for zone in zoneList:
                if site.location == zone.location:
                    site.buildOn()

        return True

    def get_zone_cost(self, zoneList):
        cost_sum = 0
        for zone in zoneList:
            cost = self.mapState[zone.location]
            try:
                cost_sum += (2 + int(cost))
            except Exception as e:
                pass

        return cost_sum
