import random

class ActionManager(object):
    #To place the troops
    def __init__(self, VectorMap, settings, GameMap):
        self.VectorMap = VectorMap
        self.settings = settings
        self.map = GameMap
    def setup(self):
        list_42 = (xrange(0, 42))
        random.shuffle(list_42)
        regions = [12]
        for i in range(0, len(regions)):
            regions[i] = list_42[i]

        #Hardcoded so the AI always goes for Australia
        regions[0] = 39
        regions[1] = 40
        regions[2] = 41
        regions[3] = 42
        return ' '.join(regions)

    #Helper function for allocate_troops
    # def isMine(self, countries,country):
    #     for i in countries:
    #         if country == i:
    #             return True
    #     return False
    def attack_transfer(self, priorities):
    #Priorities is a 82 element vector giving a float value for each region border
        if (priorities == None or len(priorities) == 0):
            priorities = [random.random() for p in range(82)]

        vm = self.VectorMap
        print(str(vm.attack_threshold))
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
                    print("found in border")
                    index = vm.borders.index([region.id, target_region.id])
                    print("found in border")
                elif [target_region.id, region.id] in vm.borders:
                    index = vm.borders.index([target_region.id, region.id])                            
                if priorities[index] > vm.attack_threshold: #Currently arbitrary threshold for transfer/attack
                    print("exceeds attack threshold")
                    actions.append(target_region.id)
            armies_per_action = (region.troop_count - 1)/len(actions) #Split up armies equally; can change this to be based on priority
            
        for action in actions:
            attack_transfers.append([region.id, action, armies_per_action])

        region.troop_count -= armies_per_action * len(actions) #Not necessarily just 1 because of integer division         
        if len(attack_transfers) == 0:
            return 'No moves'  
        return ', '.join(['%s attack/transfer %s %s %s' % (self.settings['your_bot'], attack_transfer[0],
       attack_transfer[1], attack_transfer[2]) for attack_transfer in attack_transfers])

    # def allocate_troops(self, soft_max_countries, num_troops, countries):
    # #Given a list of countries and a number of troops, 
    # #How many troops to each country depending on a soft-max function
    #     amount_troops = [[2]]
    #     amount_troops.append([42])
    #     allocate_troops = []
    #     troops = num_troops
    #     for i in range(0, len(soft_max_countries)):
    #         if isMine(countries, countries[i]):
    #             if soft_max_countries[i] > 0.69:
    #                 #Getting a list of countries that need the troops the most
    #                 allocate_troops.append(i)
    
    #     #We sort them so the countries that need the troops the most get troops first
    #     allocate_troops.sort()
    #     if (len(allocate_troops) >= 2):
    #         countries[allocate_troops[0] + 41] += 2
    #         #+41 because the second half is for how many troops there are
    #         amount_troops[0].append(2)
    #         amount_troops[1].append(allocate_troops[0])
    #         #keeping track of the how many and where we place the troops
    #         countries[allocate_troops[1] + 41] += 1
    #         amount_troops[0].append(1)
    #         amount_troops[1].append(allocate_troops[1])
    #         troops -= 3
    #         while troops > 0:
    #             for i in range(0, len(allocate_troops)):
    #                 if troops == 0:
    #                     break
    #                 countries[allocate_troops[i] + 41] += 1
    #                 if i in range(0, len(amount_troops[0])):
    #                     amount_troops[0].append(1)
    #                     amount_troops[1].append(i)
    #                 else:
    #                     amount_troops[0][i] += 1

    #                 troops -=1
                
    #     else:
    #         #Only one place was important enough to place
    #         countries[allocate_troops[0]] += troops
    #         amount_troops[0].append(troops)
    #         amount_troops[1].append(allocate_troops[0])
    #         #', '.join('%s place_armies %s %d' % ('your_bot', amount_troops[1][i], amount_troops[0][i]) for i in range(0, len(amount_troops)))
    #         #Use the above code in the main bot.py
    #     return (amount_troops) 
