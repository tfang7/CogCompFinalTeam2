class VectorMap():
	def __init__(self):
		self.MapData = []
		self.numRegions = 42
		
	def setup(self, GameMap, player, enemy):
		self.RegionOwners = []
		self.RegionArmies = []
		self.player = player
		self.enemy = enemy
		for r in range(self.numRegions):
			self.RegionOwners.append(0)
			self.RegionArmies.append(0)
		self.readMap(GameMap)

	def readMap(self, GameMap):
		regions = GameMap.regions
		for region in regions:
			rID = int(region.id)-1
			self.RegionOwners[rID] = self.enumRegion(region.owner)
			self.RegionArmies[rID] = region.troop_count

	def readRegion(self, regionID, regionOwner, regionArmy):
		rID = int(regionID)-1
		self.RegionOwners[rID] = self.enumRegion(regionOwner)
		self.RegionArmies[rID] = regionArmy

	def enumRegion(self, regionOwner):
		res = 0
		if regionOwner == self.player:
			res = 1
		elif regionOwner == self.enemy:
			res = -1
		return res

	def data(self):
		data = []
		for i in range(self.numRegions * 2):
			data.append(0)
		for j in range(self.numRegions):
			data[j] = self.RegionOwners[j]
			data[j+self.numRegions] = self.RegionArmies[j]
		return data

	def getRegions(self):
		out = ""
		for i in range(self.numRegions):
			out += str(self.RegionOwners[i]) + " "
			if (i+1)%7==0:
				out += "\n"
		return out

	def getArmies(self):
		out = ""
		for i in range(self.numRegions):
			out += str(self.RegionArmies[i]) + " "
			if (i+1)%7==0:
				out += "\n"
		return out


def main():
	VectorMap()
