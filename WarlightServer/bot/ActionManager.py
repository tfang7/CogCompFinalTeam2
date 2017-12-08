import random
class ActionManager(object):
    #To place the troops
    def __init__(self, VectorMap, settings, GameMap):
        self.VectorMap = VectorMap
        self.settings = settings
        self.map = GameMap
        self.soft_map = []
    def set_soft_map(self):
       i = 0

       while(i < 42):
           self.soft_map.append(random.uniform(0, 1))

    def setup(self):
        list_42 = (xrange(0, 42))
        random.shuffle(list_42)
        regions = []
        for i in range(0, len(regions)):
            regions.append(list_42[i])

        #Hardcoded so the AI always goes for Australia
        regions[0] = 39
        regions[1] = 40
        regions[2] = 41
        regions[3] = 42
        return ' '.join(regions)

    #Helper function for allocate_troops
    def attack_transfer(self, priorities):
    #Priorities is a 82 element vector giving a float value for each region border
        vm = self.VectorMap
        attack_transfers = [] #List to be returned
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        for region in owned_regions:
            neighbours = list(region.neighbours)
        actions = [] #The actions that we want to take. If there are multiple, split up armies to each.
        for target_region in neighbours:
            #Look up the border pair in VectorMap. Could be reversed, so check both orientations. 
            if [region.id, target_region.id] in vm.borders:
                i = vm.borders.index([region.id, target_region.id])
            elif [target_region.id, region.id] in vm.borders:
                i = vm.borders.index([target_region.id, region.id])                            
            if priorities[i] > vm.attack_threshold: #Currently arbitrary threshold for transfer/attack
                actions.append(target_region.id)
                    
            armies_per_action = (region.troop_count - 1)/len(actions) #Split up armies equally; can change this to be based on priority
            for action in actions:
                attack_transfers.append([region.id, action, armies_per_action])
            region.troop_count -= armies_per_action * len(actions) #Not necessarily just 1 because of integer division         
        if len(attack_transfers) == 0:
            return 'No moves'  
        return ', '.join(['%s attack/transfer %s %s %s' % (self.settings['your_bot'], attack_transfer[0],
       attack_transfer[1], attack_transfer[2]) for attack_transfer in attack_transfers])

    def allocate_troops(self, num_troops):
    #Given a list of countries and a number of troops, 
    #How many troops to each country depending on a soft-max function
        #priority/sum priorities * # troops floor 
        self.set_soft_map()
        amount_troops = [[]]
        amount_troops.append([])

        troops = num_troops
        sum_soft = sum(self.soft_map)
        for i in range(0, len(self.soft_map)):
            if (self.VectorMap.RegionData[str(i + 1)]["owner"] == 1):
                amount_troops[0].append(floor((soft_max_countries[i]/sum_soft) * num_troops))
                amount_troops[1].append(i)

        return (amount_troops) 
