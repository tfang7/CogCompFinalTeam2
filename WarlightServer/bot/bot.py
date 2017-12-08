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

from sys import stderr, stdin, stdout
from VectorMap import VectorMap
from GameData import *
from ActionManager import ActionManager

class Bot(object):
    '''
    Main bot class
    '''
    def __init__(self):
        '''
        Initializes a map instance and an empty dict for settings
        '''
        f = open("regions.txt", "w")
        f.write("")
        f.close()
        self.newGame = False
        self.gamesPlayed = 0

        self.VectorMap = VectorMap()
        self.settings = {}
        self.map = Map()
        self.ActionManager = ActionManager(self.VectorMap, self.settings, self.map)

    def readFromServer(self):
        #pass 84 vector into NN
        pass
    def WriteFromServer(self):
        #obtain output from NN        
        pass
    def OnGameStart(self):
        self.newGame = False
        #neural network
    def OnGameEnd(self):
        print("game over")
        self.newGame = True
        self.gamesPlayed += 1
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
                    self.OnGameStart()

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
                elif command == "opponent_moves":
                    pass
                elif command == "GAME_OVER":
                    self.OnGameEnd()
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
        self.VectorMap.setup(self.map, self.settings['your_bot'], self.settings['opponent_bot'])

    def update_map(self, options):
        '''
        Method to update our map every round.
        '''
        vm = self.VectorMap
        for i in range(0, len(options), 3):
            region = self.map.get_region_by_id(options[i])
            region.owner = options[i + 1]
            region.troop_count = int(options[i + 2])
            vm.readRegion(options[i], region.owner, region.troop_count)


        f = open("regions.txt", "a")
        output = ("Games Played: " + str(self.gamesPlayed) + "\nTensor Data\n" )
        output += (vm.printTensor(vm.createTensor()))
        output += "\n"

        # output += ("Ally Data\n" )
        # output += (self.VectorMap.getRegionData("owner"))
        # output += "\n"
        f.write(output)
        f.close()

        
    def pick_starting_regions(self, options):
        '''
        Method to select our initial starting regions.
        
        Currently selects six random regions.
        '''
        #return (str(self.ActionManager.setup()[:6]))
        shuffled_regions = Random.shuffle(Random.shuffle(options))
        return str(shuffled_regions[:6])#self.ActionManager.setup()

    def place_troops(self):

		amount_troops = self.ActionManager.allocate_troops(self.settings['starting_armies'])    
		output = ""
		placements = []
		for key in amount_troops.keys():
			tmp = [key, amount_troops[key]]
			placements.append(tmp)

		return ', '.join(['%s place_armies %s %d' % (self.settings['your_bot'], placement[0],
            placement[1]) for placement in placements])

    def attack_transfer(self):
        PrioritiesFromNeuralNetwork = None
        attack = self.ActionManager.attack_transfer(PrioritiesFromNeuralNetwork)
        return attack
        
if __name__ == '__main__':
    Bot().run()