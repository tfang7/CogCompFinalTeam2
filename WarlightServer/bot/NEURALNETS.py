import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
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

class experience_buffer():
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

def mysend(sock, msg, msglen):
    totalsent = 0

    padding = (2400-msglen)*' '
    msg += padding

    while totalsent < 2400:
        sent = sock.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent

def myreceive(sock,msglen):
    chunks = []
    bytes_recd = 0
    while bytes_recd < 2400:
        chunk = sock.recv(min(2400 - bytes_recd, 2048))
        if chunk == '':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    return ''.join(chunks)

def main():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    host = socket.gethostname()                           
    port = 6998
    serversocket.bind((host, port))                                  
    serversocket.listen(1)                                           
    clientsocket,addr = serversocket.accept()      
    print("Got a connection from %s" % str(addr))

    #Setting the training parameters
    batch_size = 32 #How many experience traces to use for each training step.
    trace_length = 8 #How long each experience trace will be when training
    update_freq = 5 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    annealing_steps = 100 #How many steps of training to reduce startE to endE.
    num_episodes = 10 #How many episodes of game environment to train network with.
    pre_train_steps = 100 #How many steps of random actions before training begins.
    load_model = False #Whether to load a saved model.
    path = "./drqn" #The path to save our model to.
    max_epLength = 100 #The max allowed length of our episode.
    time_per_step = 1 #Length of each step used in gif creation
    summaryLength = 100 #Number of epidoes to periodically save for analysis
    tau = 0.001

    tf.reset_default_graph()

    #create networks
    mainPN = PlaceNetwork()
    mainFN = FightNetwork()
    targetPN = PlaceNetwork()
    targetFN = FightNetwork()

    #initialize everything
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables,tau)
    myBuffer = experience_buffer()

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE)/annealing_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            episodeBuffer = experience_buffer()
            
            s = pickle.loads(myreceive(clientsocket,2400))
            print(s)

            d = False
            rAll = 0
            j = 0
            
            #The Q-Networks
            while j < max_epLength: 
                j+=1
                
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a1 = np.random.randint(0,41)
                    place = np.random.rand(1,42)[0]
                else:
                    a1 = sess.run(mainPN.predict,feed_dict={mainPN.X:[s]})[0]
                    place = sess.run(mainPN.Qout,feed_dict={mainPN.X:[s]})[0]
                
                place_pick = pickle.dumps(place)
                mysend(clientsocket,place_pick,len(place_pick))
                
                
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a2 = np.random.randint(0,81)
                    border = np.random.rand(1,82)[0]
                else:
                    a2 = sess.run(mainFN.predict,feed_dict={mainFN.X:[s]})[0]
                    border = sess.run(mainFN.Qout,feed_dict={mainFN.X:[s]})[0]
                
                border_pick = pickle.dumps(border)
                mysend(clientsocket,border_pick,len(border_pick))

                s1 = pickle.loads(myreceive(clientsocket,2400))
                r = pickle.loads(myreceive(clientsocket,2400))[0]
                print(s1, r)
                total_steps += 1
                
                #Save the experience to our episode buffer.
                episodeBuffer.add(np.reshape(np.array([s,a1,a2,r,s1,d]),[1,6])) 
                
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        #Get a random batch of experiences.
                        trainBatch = myBuffer.sample(batch_size)
                        
                        #Below we perform the Double-DQN update to the target Q-values
                        P1 = sess.run(mainPN.predict,feed_dict={mainPN.X:np.vstack(trainBatch[:,4])})
                        P2 = sess.run(targetPN.Qout,feed_dict={targetPN.X:np.vstack(trainBatch[:,4])})
                        F1 = sess.run(mainFN.predict,feed_dict={mainFN.X:np.vstack(trainBatch[:,4])})
                        F2 = sess.run(targetFN.Qout,feed_dict={targetFN.X:np.vstack(trainBatch[:,4])})
                        
                        end_multiplier = -(trainBatch[:,5] - 1)
                        
                        doubleP = P2[range(batch_size),P1]
                        targetP = trainBatch[:,3] + (y*doubleP * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainPN.updateModel,feed_dict={mainPN.X:np.vstack(trainBatch[:,0]),
                                                                   mainPN.targetQ:targetP, 
                                                                   mainPN.actions:trainBatch[:,1]})
                        
                        doubleF = F2[range(batch_size),F1]
                        targetF = trainBatch[:,3] + (y*doubleF * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainFN.updateModel,feed_dict={mainFN.X:np.vstack(trainBatch[:,0]),
                                                                   mainFN.targetQ:targetF, 
                                                                   mainFN.actions:trainBatch[:,2]})
                        updateTarget(targetOps,sess) #Update the target network toward the primary network.
                        
                rAll += r
                s = s1
                d = pickle.loads(myreceive(clientsocket,2400))[0]
                if d == False:
                    break
            
            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(rAll)

if __name__ == "__main__":
    main()