import random
import math
import numpy as np

class ActionManager(object):
    #To place the troops
    def __init__(self, VectorMap, settings, GameMap):
        self.VectorMap = VectorMap
        self.settings = settings
        self.map = GameMap
        self.numRegions = 42

    def pick_starting_regions(self):
        list_42 = [i for i in range(self.numRegions)]
        random.shuffle(list_42)
        regions = [i for i in range(self.numRegions)]
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
        vm = self.VectorMap
        #print(str(vm.attack_threshold))
        attack_transfers = [] #List to be returned
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        already_acting = []
        
        for region in owned_regions:
            #check if region is lameduck
            if region.troop_count==1:
                continue
            neighbours = list(region.neighbours)
            action = -1 #The actions that we want to take. If there are multiple, then somebody didn't think through the logic properly
            index = 0
            best_priority = 0 #finds best action for a given region
            for target_region in neighbours:
                #Look up the border pair in VectorMap. Could be reversed, so check both orientations. 
                if [region.id, target_region.id] in vm.borders:
                    index = vm.borders.index([region.id, target_region.id])
                elif [target_region.id, region.id] in vm.borders:
                    index = vm.borders.index([region.id, target_region.id])
                
                #transfer case
                if target_region in owned_regions:
                    #if we have already covered this relationship
                    if not (priorities[index] > vm.attack_threshold and priorities[index] > best_priority):
                        continue
                    if target_region.troop_count == 1 and region.troop_count > 1:
                        best_priority = priorities[index]
                        action = target_region.id
                    if target_region.troop_count > 1 and region.troop_count > 1:
                        #add logic here to determine best place to transfer
                        #don't double count
                        if target_region.id < region.id:
                            continue
                        best_priority = priorities[index]
                        action = target_region.id
                elif priorities[index] > vm.attack_threshold and priorities[index] > best_priority:
                    action = target_region.id
                    best_priority = priorities[index]
            
            if action != -1:
                attack_transfers.append([region.id, action, region.troop_count])
        if len(attack_transfers) == 0:
            return 'No moves'  
        return ', '.join(['%s attack/transfer %s %s %s' % (self.settings['your_bot'], attack_transfer[0], attack_transfer[1], attack_transfer[2]) for attack_transfer in attack_transfers])

    def allocate_troops(self, num_troops, priorities):
    #Given a list of countries and a number of troops, 
    #How many troops to each country depending on a soft-max function
        #priority/sum priorities * # troops floor 
        amount_troops = {}
        troops = int(num_troops)
        print("TROOP COUNT:" + str(num_troops))
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        print("LENGTH OF OWNED REGIONS: " + str(len(owned_regions)))
        sum_ownership = sum([priorities[int(r.id)-1] for r in owned_regions])

        new_priorities = [i/sum_ownership for i in priorities]
        troops_allocated = 0
        maxVal = 0
        maxID = 0
        for r in owned_regions:
            alloc = float(priorities[int(r.id)-1]) * float(num_troops)
            print("RID: " +  str(alloc) + " | " + str(int(alloc)))
            amount_troops[int(r.id)] = int(alloc)
            troops_allocated += int(alloc)
            if (alloc > maxVal):
                maxVal = alloc
                maxID = int(r.id)
        print("AMOUNT TROOPS:" + str(amount_troops))
        print("TROOPS ALLOCATED:" + str(troops_allocated))
        amount_troops[maxID] += (troops - troops_allocated)
        output = ""
        placements = []
        for key in amount_troops.keys():
            tmp = [key, amount_troops[key]]
            if (amount_troops[key] > 0):
                placements.append(tmp)
        print(', '.join(['%s place_armies %s %d' % (self.settings['your_bot'], placement[0], placement[1]) for placement in placements]))
        return ', '.join(['%s place_armies %s %d' % (self.settings['your_bot'], placement[0], placement[1]) for placement in placements])