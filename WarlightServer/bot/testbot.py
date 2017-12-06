#---------------------------------------------------------------------#
# Warlight AI Challenge - Starter Bot                                 #
# ============                                                        #
#                                                                     #
# Last update: 20 Mar, 2014                                           #
#                                                                     #
# @author Jackie <jackie@starapple.nl>                                #
# @version 1.0                                                        #
# @license MIT License (http://opensource.org/licenses/MIT)           # 
#---------------------------------------------------------------------#

from math import fmod, pi
from sys import stderr, stdin, stdout
from time import clock

class Bot(object):
    '''
    Main bot class
    '''
    def __init__(self):
        '''
        Initializes a map instance and an empty dict for settings
        '''
        self.settings = {}
        self.map = Map()

    def run(self):
        '''
        Main loop
        
        Keeps running while being fed data from stdin.
        Writes output to stdout, remember to flush!
        '''
        while not stdin.closed:
            try:
                rawline = stdin.readline()

                # End of file check
                if len(rawline) == 0:
                    break

                line = rawline.strip()

                # Empty lines can be ignored
                if len(line) == 0:
                    continue

                parts = line.split()

                command = parts[0]

                # All different commands besides the opponents' moves
                if command == 'settings':
                    self.update_settings(parts[1:])

                elif command == 'setup_map':
                    self.setup_map(parts[1:])

                elif command == 'update_map':
                    self.update_map(parts[1:])

                elif command == 'pick_starting_regions':
                    stdout.write(self.pick_starting_regions(parts[2:]) + '\n')
                    stdout.flush()

                elif command == 'go':

                    sub_command = parts[1]

                    if sub_command == 'place_armies':

                        stdout.write(self.place_troops() + '\n')
                        stdout.flush()

                    elif sub_command == 'attack/transfer':

                        stdout.write(self.attack_transfer() + '\n')
                        stdout.flush()

                    else:
                        stderr.write('Unknown sub command: %s\n' % (sub_command))
                        stderr.flush()
                elif command == 'opponent_moves':
                    pass
                else:
                    stderr.write('Unknown command: %s\n' % (command))
                    stderr.flush()
            except EOFError:
                return
    
    def update_settings(self, options):
        '''
        Method to update game settings at the start of a new game.
        '''
        key, value = options
        self.settings[key] = value

    def setup_map(self, options):
        '''
        Method to set up essential map data given by the server.
        '''
        map_type = options[0]

        for i in range(1, len(options), 2):

            if map_type == 'super_regions':

                super_region = SuperRegion(options[i], int(options[i + 1]))
                self.map.super_regions.append(super_region)

            elif map_type == 'regions':

                super_region = self.map.get_super_region_by_id(options[i + 1])
                region = Region(options[i], super_region)
                
                self.map.regions.append(region)
                super_region.regions.append(region)

            elif map_type == 'neighbors':

                region = self.map.get_region_by_id(options[i])
                neighbours = [self.map.get_region_by_id(region_id) for region_id in options[i + 1].split(',')]

                for neighbour in neighbours:
                    region.neighbours.append(neighbour)
                    neighbour.neighbours.append(region)

        if map_type == 'neighbors':
            
            for region in self.map.regions:

                if region.is_on_super_region_border:
                    continue

                for neighbour in region.neighbours:

                    if neighbour.super_region.id != region.super_region.id:

                        region.is_on_super_region_border = True
                        neighbour.is_on_super_region_border = True

    def update_map(self, options):
        '''
        Method to update our map every round.
        '''
        for i in range(0, len(options), 3):
            
            region = self.map.get_region_by_id(options[i])
            region.owner = options[i + 1]
            region.troop_count = int(options[i + 2])
            
    def pick_starting_regions(self, options):
        '''
        Method to select our initial starting regions.
        
        Currently selects six random regions.
        '''
        shuffled_regions = Random.shuffle(Random.shuffle(options))
        
        return ' '.join(shuffled_regions[:6])

    def place_troops(self):
        '''
        Method to place our troops.
        
        Currently keeps places a maximum of two troops on random regions.
        '''
        placements = []
        region_index = 0
        troops_remaining = int(self.settings['starting_armies'])
        
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        duplicated_regions = owned_regions * (3 + int(troops_remaining / 2))
        shuffled_regions = Random.shuffle(duplicated_regions)
        
        while troops_remaining:

            region = shuffled_regions[region_index]
            
            if troops_remaining > 1:

                placements.append([region.id, 2])

                region.troop_count += 2
                troops_remaining -= 2
                
            else:

                 placements.append([region.id, 1])

                 region.troop_count += 1
                 troops_remaining -= 1

            region_index += 1
            
        return ', '.join(['%s place_armies %s %d' % (self.settings['your_bot'], placement[0],
            placement[1]) for placement in placements])

    def attack_transfer(self):
        '''
        Method to attack another region or transfer troops to allied regions.
        
        Currently checks whether a region has more than six troops placed to attack,
        or transfers if more than 1 unit is available.
        '''
        attack_transfers = []
        
        owned_regions = self.map.get_owned_regions(self.settings['your_bot'])
        
        for region in owned_regions:
            neighbours = list(region.neighbours)
            while len(neighbours) > 0:
                target_region = neighbours[Random.randrange(0, len(neighbours))]
                if region.owner != target_region.owner and region.troop_count > 6:
                    attack_transfers.append([region.id, target_region.id, 5])
                    region.troop_count -= 5
                elif region.owner == target_region.owner and region.troop_count > 1:
                    attack_transfers.append([region.id, target_region.id, region.troop_count - 1])
                    region.troop_count = 1
                else:
                    neighbours.remove(target_region)
        
        if len(attack_transfers) == 0:
            return 'No moves'
        
        return ', '.join(['%s attack/transfer %s %s %s' % (self.settings['your_bot'], attack_transfer[0],
            attack_transfer[1], attack_transfer[2]) for attack_transfer in attack_transfers])

class Map(object):
    '''
    Map class
    '''
    def __init__(self):
        '''
        Initializes empty lists for regions and super regions.
        '''
        self.regions = []
        self.super_regions = []

    def get_region_by_id(self, region_id):
        '''
        Returns a region instance by id.
        '''
        return [region for region in self.regions if region.id == region_id][0]
    
    def get_super_region_by_id(self, super_region_id):
        '''
        Returns a super region instance by id.
        '''
        return [super_region for super_region in self.super_regions if super_region.id == super_region_id][0]

    def get_owned_regions(self, owner):
        '''
        Returns a list of region instances owned by `owner`.
        '''
        return [region for region in self.regions if region.owner == owner]

class SuperRegion(object):
    '''
    Super Region class
    '''
    def __init__(self, super_region_id, worth):
        '''
        Initializes with an id, the super region's worth and an empty lists for 
        regions located inside this super region
        '''
        self.id = super_region_id
        self.worth = worth
        self.regions = []

class Region(object):
    '''
    Region class
    '''
    def __init__(self, region_id, super_region):
        '''
        '''
        self.id = region_id
        self.owner = 'neutral'
        self.neighbours = []
        self.troop_count = 2
        self.super_region = super_region
        self.is_on_super_region_border = False

class Random(object):
    '''
    Random class
    '''
    @staticmethod
    def randrange(min, max):
        '''
        A pseudo random number generator to replace random.randrange
        
        Works with an inclusive left bound and exclusive right bound.
        E.g. Random.randrange(0, 5) in [0, 1, 2, 3, 4] is always true
        '''
        return min + int(fmod(pow(clock() + pi, 2), 1.0) * (max - min))

    @staticmethod
    def shuffle(items):
        '''
        Method to shuffle a list of items
        '''
        i = len(items)
        while i > 1:
            i -= 1
            j = Random.randrange(0, i)
            items[j], items[i] = items[i], items[j]
        return items

if __name__ == '__main__':
    '''
    Not used as module, so run
    '''
    Bot().run()