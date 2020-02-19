def get_distance(loc1, loc2):  # manhattan distance
	return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

class Site(object):
	"""docstring for Site"""

	def __init__(self, name:str, score:dict, build_on_cost:int, location:tuple):
		super(Site, self).__init__()
		self.name = name
		self.score = score
		self.backupScore = score
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
		# please call this method whenever you tried to build on to recover
		# or simply call this method anytime after calling zone.get_score
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

	def __init__(self, mapState):
		super(Map, self).__init__()
		self.mapState = mapState  # a 2D numpy matrix
		self.siteList = self.get_site_List()

	def get_site_List(self):
		# sth to implement here
		return # this should return a list of sites, each an object of class Site

	def get_score(self, zoneList): 
		self.checkBuildOn(zoneList)
		score = self.get_zone_zone_score(zoneList) + \
				self.get_zone_site_score(self.siteList,zoneList) - \
				self.get_zone_cost(zoneList)

		for site in siteList:
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
