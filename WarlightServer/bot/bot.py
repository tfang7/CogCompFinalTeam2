from sys import stderr, stdin, stdout
from VectorMap import VectorMap
from GameData import *
from ActionManager import ActionManager
import numpy as np
import socket
import pickle
import time
import random
import NeuralNets
import _thread
class Bot(object):
    '''
    Main bot class
    '''
    def __init__(self,sock=None):
        '''
        Initializes a map instance and an empty dict for settings
        '''
        f = open("regions.txt", "w")
        f.write("")
        f.close()
        self.newGame = False
        self.gamesPlayed = 0

        self.episode_turn = 0
        self.vec84 = np.zeros(84)

        self.VectorMap = VectorMap()
        self.settings = {}
        self.map = Map()
        self.ActionManager = ActionManager(self.VectorMap, self.settings, self.map)

        self.Trainer = NeuralNets.Trainer()
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # host = socket.gethostname()                           
        # port = 6998
        # self.socket.connect((host, port))  


    def OnGameEnd(self):
        print("game over")
        self.newGame = True
        self.first_send = True
        self.gamesPlayed += 1


    # def mysend(self,sock, msg, msglen):
    #     totalsent = 0

    #     padding = (2400-msglen)*' '
    #     msg += padding

    #     while totalsent < 2400:
    #         sent = sock.send(msg[totalsent:])
    #         if sent == 0:
    #             raise RuntimeError("socket connection broken")
    #         totalsent = totalsent + sent

    # def myreceive(self,sock,msglen):
    #     chunks = []
    #     bytes_recd = 0
    #     while bytes_recd < 2400:
    #         chunk = sock.recv(min(2400 - bytes_recd, 2048))
    #         if chunk == '':
    #             raise RuntimeError("socket connection broken")
    #         chunks.append(chunk)
    #         bytes_recd = bytes_recd + len(chunk)
    #     return ''.join(chunks)

    def compute_reward(self, countries,troops):
        delta_countries_owned = self.VectorMap.count_countries()-countries
        delta_troops_owned = self.VectorMap.count_troops()-troops
        return delta_countries_owned
        # return delta_troops_owned

    def run(self):
        '''
        Main loop
        
        Keeps running while being fed data from stdin.
        Writes output to stdout, remember to flush!
        '''
        
        self.first_send = True
        self.reward = 0
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
                    print("Episode Turn: {}".format(self.episode_turn))
                    self.setup_map(parts[1:])


                elif command == 'update_map':
                    print()
                    countries_owned = self.VectorMap.count_countries()
                    troops_owned = self.VectorMap.count_troops()
                    self.update_map(parts[1:])
                    tensor = np.array(self.VectorMap.createTensor())

                    if (self.episode_turn == 0):
                        self.Trainer.init_episode(tensor, self.episode_turn)
                        self.episode_turn += 1
                    else:
                        self.reward = self.compute_reward(countries_owned,troops_owned)
                        rewards = np.array([self.reward])
                        self.Trainer.train_reward(tensor, rewards)

                        

                   #  vec84 = np.array(self.VectorMap.createTensor())
                   # # send84 = pickle.dumps(np.random.random(84))
                   #  #mysend(s,send84,len(send84)) 
                    

                   #    #  send_r = pickle.dumps(np.array([self.reward]))
                   #     # mysend(s,sendr,len(send_r))
                   #  else:
                   #      self.first_send = False

                elif command == 'pick_starting_regions':
                    stdout.write(self.pick_starting_regions(parts[2:]) + '\n')
                    stdout.flush()
                    self.newGame = False

                elif command == 'go':
                    self.episode_turn += 1
                    sub_command = parts[1]
                    tensor = np.array(self.VectorMap.createTensor())
                    moves = self.Trainer.get_moves(self.episode_turn)
                    if sub_command == 'place_armies':
                        output = self.place_troops(moves[0])
                        stdout.write(output)
                        stdout.flush()

                    elif sub_command == 'attack/transfer':
                        output = self.attack_transfer(moves[1])
                        stdout.write(output)
                        stdout.flush()
                    else:
                        stderr.write('Unknown sub command: %s\n' % (sub_command))
                        stderr.flush()
                elif command == "opponent_moves":
                    pass
                elif command == "GAME_OVER":
                    self.OnGameEnd()
                    continue
                else:
                    stderr.write('Unknown command: %s\n' % (command))
                    stderr.flush()
            except EOFError:
                return
           # send_ng = pickle.dumps(np.array([self.newGame]))
            #mysend(s,sendr,len(send_ng))

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
        return self.ActionManager.pick_starting_regions()

    def place_troops(self,priorities42):
        place = self.ActionManager.allocate_troops(self.settings['starting_armies'], priorities42)    
        return place

    def attack_transfer(self,priorities82):
        attack = self.ActionManager.attack_transfer(priorities82)
        return attack
        
if __name__ == '__main__':
    Bot().run()