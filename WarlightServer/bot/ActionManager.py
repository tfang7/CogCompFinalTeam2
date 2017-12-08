import random
import math


class ActionManager(object):
    #To place the troops
    def __init__(self, VectorMap, settings, GameMap):
        self.VectorMap = VectorMap
        self.settings = settings
        self.map = GameMap
        self.soft_map = [random.random() for soft in range(42)]
    def setup(self):
        list_42 = (xrange(0, 42))
        random.shuffle(list_42)
        regions = [12]
        for i in range(0, len(regions)):
            regions[i] = list_42[i]

        #Hardcoded so the AI always goes for Australia
        regions[1] = 39
        regions[2] = 40
        regions[3] = 41
        regions[4] = 42
        return ' '.join(str(regions))

    #Helper function for allocate_troops
    def attack_transfer(self, priorities):
    #Priorities is a 82 element vector giving a float value for each region border
        if (priorities == None or len(priorities) == 0):
            priorities = [random.random() for p in range(82)]

        vm = self.VectorMap
        #print(str(vm.attack_threshold))
        attack_transfers = [] #List to be returned
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        armies_per_action = 0
        for region in owned_regions:
            neighbours = list(region.neighbours)
            actions = [] #The actions that we want to take. If there are multiple, split up armies to each.
            index = 0
            for target_region in neighbours:
                #Look up the border pair in VectorMap. Could be reversed, so check both orientations. 
                if [region.id, target_region.id] in vm.borders:
                    #print("found in border")
                    index = vm.borders.index([region.id, target_region.id])
                    #print("found in border")
                elif [target_region.id, region.id] in vm.borders:
                    index = vm.borders.index([target_region.id, region.id])                            
                if priorities[index] > vm.attack_threshold: #Currently arbitrary threshold for transfer/attack
                    #print("exceeds attack threshold")
                    actions.append(target_region.id)
            armies_per_action = (region.troop_count - 1)/len(actions) if len(actions) > 0 else 0 #Split up armies equally; can change this to be based on priority
            
        for action in actions:
            attack_transfers.append([region.id, action, armies_per_action])

        region.troop_count -= armies_per_action * len(actions) #Not necessarily just 1 because of integer division         
        if armies_per_action == 0:
            return 'No moves'  
        return ', '.join(['%s attack/transfer %s %s %s' % (self.settings['your_bot'], attack_transfer[0],
       attack_transfer[1], attack_transfer[2]) for attack_transfer in attack_transfers])

    def allocate_troops(self, num_troops):
    #Given a list of countries and a number of troops, 
    #How many troops to each country depending on a soft-max function
        #priority/sum priorities * # troops floor 
        amount_troops = {}

        troops = int(num_troops)
        sum_soft = sum(self.soft_map)
        highest_priority = 0
        for i in range(0, len(self.soft_map)):
            if (self.VectorMap.RegionData[str(i + 1)]["owner"] == 1):
                amount_troops[str(i + 1)] = (math.floor(float(num_troops) * (self.soft_map[i]/float(sum_soft))))
                troops -= math.floor(float(num_troops) * (self.soft_map[i]/float(sum_soft)))
                if self.soft_map[i] > self.soft_map[highest_priority]:
                    highest_priority = i
        if (troops > 0):
            troops_addition = int(amount_troops[str(highest_priority + 1)])
            amount_troops[str(highest_priority + 1)] = troops + troops_addition
        for key in amount_troops.keys():
            if amount_troops[key] == 0:
                del amount_troops[key]

        return (amount_troops) 
