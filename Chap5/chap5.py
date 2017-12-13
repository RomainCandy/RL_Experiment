#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:35:55 2017

@author: romain
"""

import random as rd
import numpy as np


class Env():
    def __init__(self):
        raise NotImplementedError('what the env looks like, \
                                  must implement the Q(s,a) representation')

    def step(self,action):
        raise NotImplementedError('return new_state and reward')
        
    def reset(self):
        raise NotImplementedError
    
    def eps_greedy(self,epsilon, verbose = False):
        """ return an action using eps_greedy policy"""
        if rd.random()<1-epsilon:
            greedy = np.argwhere(self.Q[self.state]== np.max(self.Q[self.state]))
#            print('state ', self.state)
            return rd.choice(greedy.flatten())
        else:
            if verbose:
                print('pas greedy',self.state)
            return rd.randint(0,(self.Q[self.state].shape)[0]-1)
        
    def start(self):
        raise NotImplementedError
    
    def is_finished(self):
        raise NotImplementedError
        
    def get_episode(self,epsilon=.05, verbose = False):
        states_actions = list()
        rewards = list()
        self.start()
        while True:
            action = self.eps_greedy(epsilon, verbose)
            new_state,reward = self.step(action)
            try:
                states_actions.append(tuple(list(self.state)+[action]))
            except TypeError:
                
                states_actions.append(tuple([self.state]+[action]))
            rewards.append(reward)
            if new_state in self.terminal_states:
                break
            else:
                self.state = new_state
        return states_actions,rewards
    
    def monte_carlo(self,nb_iter = 1000,epsilon = .05,discount = 1,first_visit=True):
        self.reset()
        ret = dict()
        for i in range(nb_iter):
            already_seen = set()
            states_actions,rewards = self.get_episode(epsilon)
            n = len(rewards)
            for index,state_action in enumerate(states_actions):
                if state_action not in already_seen:
                    already_seen.add(state_action)
                    G = np.sum(rewards[k]*discount**k for k in range(index,n))
                    try:
                        ret[state_action][0] += 1
                        ret[state_action][1] += (G-ret[state_action][1])/ret[state_action][0]
                    except KeyError:
                        ret[state_action] = [1,G]
                        try:
                            self.Q[state_action] = ret[state_action][1]
                        except IndexError:
                            print('probleme zzzzzzzzzzzz')
                            print(state_action)
                            return
            print('épisode ' , i ,'finies en ', n , 'étapes')
            print('--'*100)
        print('learning over')
    def Q_learning(self,nb_iter = 10, alpha = .5, discount = 1):
        self.initQ()
        for t in range(nb_iter):
            self.start()
            i = 0
            while True:
                action = self.eps_greedy(.05)
                old_state_x,old_state_y = self.state
                reward = self.step(action)
                if self.is_terminal():
                    self.Q[old_state_x,old_state_y,action] += alpha*(reward - 
                          self.Q[old_state_x,old_state_y,action])
                    break
                self.Q[old_state_x,old_state_y,action] += alpha*(reward + 
                      discount*np.max(self.Q[self.state[0],self.state[1]]) -
                      self.Q[old_state_x,old_state_y,action])
                i+=1
            print('épisode ' , t ,'finies en ', i , 'étapes')
    def get_policy(self):
        self.policy = np.argmax(self.Q,-1)
    
        

def make_circuit(file):
    with open(file) as plateau:
        res = list()
        for x in plateau.readlines():
            res.append(list(map(int,x.replace('\n','').split(','))))
        return np.array(res,dtype=int).T


class Race(Env):
    def __init__(self,circuit):
        self.circuit = make_circuit(circuit)
        self.start_line = np.argwhere(self.circuit == 2)
        self.finish_line = np.argwhere(self.circuit == 3)
        self.largeur = self.circuit.shape[0]-1
        self.hauteur = self.circuit.shape[1]-1
        self.Q = np.zeros((self.circuit.shape[0],self.circuit.shape[1],5,5,9))
        self.terminal_states = ('Terminal',)
        self.corresp_action = {0:(0,0),1:(0,-1),2:(0,1),3:(-1,0),4:(-1,-1),5:(-1,1),6:(
                1,0),7:(1,-1),8:(1,1)}
    def initQ(self):
        self.Q = np.zeros((self.circuit.shape[0],self.circuit.shape[1],5,5,9))
        
    def start(self):
        self.state = rd.choice([(x,y,0,0) for x,y in self.start_line])
        
    def reset(self):
        self.Q = np.zeros((self.circuit.shape[0],self.circuit.shape[1],5,5,9))
        
    def true_state(self,state):
        x,y,h,d = state
        if y<0 :
            return rd.choice([(x,y,0,0) for x,y in self.start_line]),-1
        elif y<len(self.finish_line):
            if x>=self.largeur:
                return (self.largeur,y,h,d),-1
            else:
                return (state),-1
        elif y>self.hauteur:
            return rd.choice([(x,y,0,0) for x,y in self.start_line]),-1
        elif self.circuit[x,y] == 0:
            return rd.choice([(x,y,0,0) for x,y in self.start_line]),-1
        else:
            return (x,y,h,d),-1
        
    def step(self,action):
        if [self.state[0],self.state[1]] in self.finish_line.tolist():
            return 'Terminal',1
        x,y,h,d = self.state
        assert h in range(-4,1)
        assert d in range(0,5)
        ah,ad = self.corresp_action[action]
        new_h = max(min(h+ah,0),-4)
        new_d = max(min(d+ad,4),0)
        new_state = x+new_d,y+new_h,new_h,new_d
        return self.true_state(new_state)

    def follow_policy(self,verbose = False):
        sa,r = self.get_episode(0.005,verbose)
        print("c'est bon lol")
        for x in sa:
            print('state = ',x[:-1])
            print('action =',x[-1])
            print('--'*50)
        
R = Race('circuit1.txt')
R.monte_carlo()
#print(R.Q.shape)
#R.start()
#print(R.largeur)
#print(R.hauteur)
#print(R.start_line)
#print(len(R.finish_line))
#print(R.state)
#R.step(5)
#print(R.finish_line)
#print(R.Q[8,31,0,0])
#R.follow_policy(True)
#R.get_policy()
#print(R.policy)

class Grid(Env):
    """grid world 
    
    0   1   2   3
    4   5   6   7
    8   9  10  11
    12 13  14  15
    """
    def __init__(self,long=3):
        self.long = long
        self.Q=np.zeros((long*long,4),float)
        self.terminal_states = ('Terminal',)
        
    def reset(self):
        self.Q=np.zeros((self.long*self.long,4))
        
    def initQ(self):
        self.Q=np.zeros((self.long**2,4),float)
    
    def start(self):
        self.state = rd.randint(1,self.long*self.long-2)
        
    def is_finished(self):
        return self.state == 'Terminal'
    
    def step(self,action):
        """ 0:S 1:N 2:E 3:W"""
        if self.state == self.long**2-1:
            return 'Terminal',50
        if self.state == 0:
            return 'Terminal',3
        if action == 0:
            if self.state - self.long<0:
                return self.state, -1
            else:
                return self.state-self.long,-1
        if action == 1:
            if self.state + self.long>=self.long**2:
                return self.state,-1
            else:
                return self.state+self.long,-1
        if action == 2:
            if self.state%self.long == self.long-1:
                return self.state,-1
            else:
                return self.state + 1,-1
        else:
            if self.state%self.long == 0:
                return self.state,-1
            else:
                return self.state - 1,-1
        
    def get_policy(self):
        Env.get_policy(self)
        corresp = {0:'S',1:'î',2:'->',3:'<-'}
        self.policy = [corresp[x] for x in self.policy]
        self.policy = np.array(self.policy).reshape(self.long,-1)    
#G = Grid(4)
#t = time.time()
#G.monte_carlo()
#print('temps écoulé ' ,time.time()-t)
#print(G.Q)
#G.get_policy()
#print(G.policy)
        
        