class VectorMap(object):
	def __init__(self):
		self.MapData = []
		self.numRegions = 42
		self.initBorders()
	def Region(self, rId, player, troops):
		return {"id": rId, "owner": player, "troops": troops}
	def initBorders(self):
		self.borders = [
			[1,2], [1,4], [1,30], [2,4], [2,3], [2,5], [3,5], [3,6], [3,14],
			[4,5], [4,7], [5,6], [5,7], [5,8], [6,8], [7,8], [7,9],
			[8,9], [9,10], [10,11], [10,12], [11,12], [11,13], [12,13], [12,21],
			[14,15], [14,16], [15,16], [15,18], [15, 19], [16,17], [17,19],
			[17,20], [17,27], [17,32], [17,36], [18,19], [18,20], [18,21],
			[19,20], [20,21], [20,22], [20,36], [21,22], [21,23], [21,24],
			[22, 23], [22,36], [23, 24], [23,25], [23,26], [23,36], [24,25], 
			[25,26], [27,28], [27,32], [27,33], [28,29], [28,31], [28,33], [28,34], [29,30], 
			[29,31], [30, 31], [30,34], [30,35], [31,34], [32,33], [32,36], [32,37],
			[33,34], [33,37], [33,38], [34,35], [36,37], [37,38], [38,39],
			[39,40], [39,41], [40,41], [40,42], [41,42]]
		
	def setup(self, GameMap, player, enemy):
		self.RegionData = {}
		self.player = player
		self.enemy = enemy
		self.readMap(GameMap)

	def readMap(self, GameMap):
		regions = GameMap.regions
		for region in regions:
			ally = self.enumRegion(region.owner)
			rID = region.id
			troops = region.troop_count
			self.RegionData[region.id] = self.Region(rID, ally, troops)

	def readRegion(self, regionID, regionOwner, regionArmy):
		r = self.RegionData[regionID]
		r["owner"] = self.enumRegion(regionOwner)
		r["troops"] = regionArmy

	def enumRegion(self, regionOwner):
		res = 0
		if regionOwner == self.player:
			res = 1
		elif regionOwner == self.enemy:
			res = -1
		return res

	def data(self):
		data = []
		return data

	def getRegionData(self, datatype):
		out = ""
		count = 0
		for rID in self.RegionData.keys():
			region = self.RegionData[rID]
			out += str(region[datatype]) + " "
			if (count+1)%7==0:
				out += "\n"
			count += 1
		return out
