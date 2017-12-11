import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import time
import itertools
import tensorflow.contrib.slim as slim
import socket
import pickle

class PlaceNetwork():
    def __init__(self):       
        self.X = tf.placeholder("float", [None, 84])
        self.W1 = tf.Variable(tf.truncated_normal([84,64],stddev=.1))
        self.B1 = tf.Variable(tf.zeros([1]))
        self.W2A = tf.Variable(tf.truncated_normal([64,32],stddev=.1))
        self.B2A = tf.Variable(tf.zeros([1]))
        self.W2V = tf.Variable(tf.truncated_normal([64,32],stddev=.1))
        self.B2V = tf.Variable(tf.zeros([1]))
        
        self.Y1 = tf.nn.relu(tf.matmul(self.X,self.W1)+self.B1)
        self.Y2A = tf.nn.relu(tf.matmul(self.Y1,self.W2A)+self.B2A)
        self.Y2V = tf.nn.relu(tf.matmul(self.Y1,self.W2V)+self.B2V)
        
        self.AW = tf.Variable(tf.truncated_normal([32,42]))
        self.VW = tf.Variable(tf.truncated_normal([32,1]))
        
        self.Advantage = tf.matmul(self.Y2A,self.AW)
        self.Value = tf.matmul(self.Y2V,self.VW)
        
        self.salience = tf.gradients(self.Advantage,self.X)
        
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,42,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class FightNetwork():
    def __init__(self):
        self.X = tf.placeholder("float", [None, 84])
        self.W1 = tf.Variable(tf.truncated_normal([84,128],stddev=.1))
        self.B1 = tf.Variable(tf.zeros([128]))
        self.W2A = tf.Variable(tf.truncated_normal([128,64],stddev=.1))
        self.B2A = tf.Variable(tf.zeros([64]))
        self.W2V = tf.Variable(tf.truncated_normal([128,64],stddev=.1))
        self.B2V = tf.Variable(tf.zeros([64]))
        
        self.Y1 = tf.nn.relu(tf.matmul(self.X,self.W1)+self.B1)
        self.Y2A = tf.nn.relu(tf.matmul(self.Y1,self.W2A)+self.B2A)
        self.Y2V = tf.nn.relu(tf.matmul(self.Y1,self.W2V)+self.B2V)
        
        self.AW = tf.Variable(tf.random_normal([64,82]))
        self.VW = tf.Variable(tf.random_normal([64,1]))
        
        self.Advantage = tf.matmul(self.Y2A,self.AW)
        self.Value = tf.matmul(self.Y2V,self.VW)
        
        self.salience = tf.gradients(self.Advantage,self.X)
        
        #same thing
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,82,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer(object):
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

class Trainer(object):
    def __init__(self):
        print("Created Trainer object\n")
        #Setting the training parameters
        self.batch_size = 32 #How many experience traces to use for each training step.
        self.trace_length = 8 #How long each experience trace will be when training
        self.update_freq = 5 #How often to perform a training step.
        self.y = .99 #Discount factor on the target Q-values
        self.startE = 1 #Starting chance of random action
        self.endE = 0.1 #Final chance of random action
        self.annealing_steps = 100 #How many steps of training to reduce startE to endE.
        self.num_episodes = 10 #How many episodes of game environment to train network with.
        self.pre_train_steps = 100 #How many steps of random actions before training begins.
        self.load_model = False #Whether to load a saved model.
        self.path = "./drqn" #The path to save our model to.
        self.max_epLength = 100 #The max allowed length of our episode.
        self.time_per_step = 1 #Length of each step used in gif creation
        self.summaryLength = 100 #Number of epidoes to periodically save for analysis
        self.tau = 0.001
        tf.reset_default_graph()
        #create networks
        self.mainPN = PlaceNetwork()
        self.mainFN = FightNetwork()
        self.targetPN = PlaceNetwork()
        self.targetFN = FightNetwork()

        #initialize everything
        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(self.trainables, self.tau)
        self.myBuffer = experience_buffer()

        #create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []
        self.total_steps = 0
        #Set the rate of random action decrease. 
        self.e = self.startE
        self.stepDrop = (self.startE - self.endE)/self.annealing_steps
        self.sess = None

    def run(self, inputTensor):
        print("Starting Session")
        with tf.Session() as self.sess:
                self.sess.run(self.init)
                for i in range(self.num_episodes):
                    episodeBuffer = experience_buffer()
                    #data = myreceive(clientsocket,2400)
                   # s = pickle.loads(myreceive(clientsocket,2400))
                   # print(s)

                    d = False
                    rAll = 0
                    j = 0
                    #s = input vector
                    #The Q-Networks
                    while j < self.max_epLength: 
                        j+=1
                        
                        #Choose an action by greedily (with e chance of random action) from the Q-network
                        if np.random.rand(1) < self.e or self.total_steps < self.pre_train_steps:
                            a1 = np.random.randint(0,41)
                            place = np.random.rand(1,42)[0]
                        else:
                            a1 = self.sess.run(self.mainPN.predict,feed_dict={self.mainPN.X:[s]})[0]
                            place = self.sess.run(self.mainPN.Qout,feed_dict={self.mainPN.X:[s]})[0]
                        
                       # place_pick = pickle.dumps(place)
                        #mysend(data)
                        
                        
                        if np.random.rand(1) < e or self.total_steps < self.pre_train_steps:
                            a2 = np.random.randint(0,81)
                            border = np.random.rand(1,82)[0]
                        else:
                            a2 = self.sess.run(self.mainFN.predict,feed_dict={self.mainFN.X:[s]})[0]
                            border = self.sess.run(self.mainFN.Qout,feed_dict={self.mainFN.X:[s]})[0]
                        
                        #border_pick = pickle.dumps(border)
                        #mysend(data)

                       # s1 = pickle.loads(myreceive(clientsocket,2400))
                       # r = pickle.loads(myreceive(clientsocket,2400))[0]
                        print(s1, r)
                        self.total_steps += 1
                        
                        #Save the experience to our episode buffer.
                        episodeBuffer.add(np.reshape(np.array([s,a1,a2,r,s1,d]),[1,6])) 
                        
                        if self.total_steps > self.pre_train_steps:
                            if self.e > self.endE:
                                self.e -= self.stepDrop
                            
                            if self.total_steps % (update_freq) == 0:
                                #Get a random batch of experiences.
                                trainBatch = self.myBuffer.sample(self.batch_size)
                                
                                #Below we perform the Double-DQN update to the target Q-values
                                P1 = sess.run(self.mainPN.predict,feed_dict={self.mainPN.X:np.vstack(trainBatch[:,4])})
                                P2 = sess.run(self.targetPN.Qout,feed_dict={self.targetPN.X:np.vstack(trainBatch[:,4])})
                                F1 = sess.run(self.mainFN.predict,feed_dict={self.mainFN.X:np.vstack(trainBatch[:,4])})
                                F2 = sess.run(self.targetFN.Qout,feed_dict={self.targetFN.X:np.vstack(trainBatch[:,4])})
                                
                                end_multiplier = -(trainBatch[:,5] - 1)
                                
                                doubleP = P2[range(batch_size),P1]
                                targetP = trainBatch[:,3] + (y*doubleP * end_multiplier)
                                #Update the network with our target values.
                                _ = sess.run(self.mainPN.updateModel,feed_dict={self.mainPN.X:np.vstack(trainBatch[:,0]),
                                                                           self.mainPN.targetQ:targetP, 
                                                                           self.mainPN.actions:trainBatch[:,1]})
                                
                                doubleF = F2[range(batch_size),F1]
                                targetF = trainBatch[:,3] + (y*doubleF * end_multiplier)
                                #Update the network with our target values.
                                _ = sess.run(self.mainFN.updateModel,feed_dict={self.mainFN.X:np.vstack(trainBatch[:,0]),
                                                                           self.mainFN.targetQ:targetF, 
                                                                           self.mainFN.actions:trainBatch[:,2]})
                                updateTarget(targetOps,self.sess) #Update the target network toward the primary network.
                                
                        rAll += r
                        s = s1
                        d = False
                       # d = pickle.loads(myreceive(clientsocket,2400))[0]
                        if d == False:
                            break
                    
                    self.myBuffer.add(episodeBuffer.buffer)
                    self.jList.append(j)
                    self.rList.append(rAll)
        return 1
