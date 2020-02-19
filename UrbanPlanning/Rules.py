from InstanceTemplate import *

def get_distance(loc1, loc2):  # manhattan distance
	return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def countZone(zoneType, zoneList):
	return sum([zone.name == zoneType for zone in zoneList])

def ifValidZoneList(zoneList, maxIndustrial, maxCommercial, maxIndustrial):
	# check if it exceed limit
	if countZone(zoneType="I",zoneList) > maxIndustrial:
		return False
	if countZone(zoneType="R",zoneList) > maxResidential:
		return False
	if countZone(zoneType="C",zoneList) > maxCommercial:
		return False

	# check if two zone at same location
	length = len(zoneList)
	for index1 in range(length):
		for index2 in range(index1,length):
			if zoneList[index1].location == zoneList[index2].location:
				return False

	return True

class Site(object):
	"""docstring for Site"""

	def __init__(self, name:str, score:dict, build_on_cost:int, location:tuple):
		super(Site, self).__init__()
		self.name = name
		self.score = {**score}
		self.backupScore = {**score}
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
		self.score = {**self.backupScore}
		return True

class Zone(object):
	"""docstring for zone"""

	def __init__(self, name: str, score: dict, location: tuple):
		super(Zone, self).__init__()
		self.name = name
		self.score = score
		self.location = location

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
				state = self.mapState[row,col]
				if state in ["X","S"]:
					siteList.append(createSite(siteName=state, location=(row,col)))
		return siteList

	def get_score(self, zoneList): # this is all you need
		self.checkBuildOn(zoneList)
		score = self.get_zone_zone_score(zoneList) + \
				self.get_zone_site_score(self.siteList,zoneList) - \
				self.get_zone_cost(zoneList)

		for site in self.siteList:
			site.recoverSite()
		return score # the larger the better!!!

	def get_zone_zone_score(self, zoneList):
		return sum([sum([zone1.get_score(zone2) for zone2 in zoneList] for zone1 in zoneList)])

	def get_zone_site_score(self, zoneList):
		return sum([sum([site.get_score(zone) for zone in zoneList]) for site in self.siteList])

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
