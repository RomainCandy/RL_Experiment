#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:51:05 2017

@author: romain
"""


import random as rd
import numpy as np

class Gridworld:
    def __init__(self,height,width,termial_states):
        self.height = height
        self.width = width
        self.terminal_states = termial_states
        self.actions = ['N','S','E','W']
        self.state = [rd.randint(0,height-1),rd.randint(0,width-1)]
        self.reward = 0
        self.V = np.zeros((height,width))
        
    def step(self,action):
        raise NotImplementedError
        
    def get_reward(self):
        raise NotImplementedError
        
        
class Ex2(Gridworld):
    def __init__(self):
        Gridworld.__init__(self,5,5,None)
        self.special_state = {(1,0):((1,4),10),(3,0):((3,2),5)}
    
    def ingrid_world(self):
        if self.state[0]< 0:
            self.state[0] = 0
            self.reward = -1
        if self.state[0]>= self.height:
            self.state[0] = self.height-1
            self.reward = -1
        if self.state[1] < 0:
            self.state[1] = 0
            self.reward = -1
        if self.state[1] >= self.width:
            self.state[1] = self.width -1
            self.reward = -1
        
    def step(self,action):
        assert action in self.actions
        if tuple(self.state) in self.special_state:
            self.state,self.reward=self.special_state[tuple(self.state)]
            self.state = list(self.state)
        else:
            self.reward = 0
            if action == 'N':
                self.state[0] -= 1
            elif action == 'S':
                self.state[0] += 1
            elif action == 'E':
                self.state[1] += 1
            else:
                self.state[1] -= 1
            self.ingrid_world()
            
    
    def __str__(self):
        return str(self.state)
    
class Policy:
    def __init__(self,actions):
        self.actions = actions
        
    def take_action(self):
        return rd.choice(self.actions)
    

class TD0:
    def __init__(self,grid,policy,discount):
        self.grid = grid
        self.policy = policy
        self.discount = discount
        self.time = 0
        
    def update(self):
        self.time += 1
        current_state = self.grid.state
        action = self.policy.take_action()
#        print('before action ',current_state,action)
        
        self.grid.step(action)
        next_state = self.grid.state
        reward = self.grid.reward
#        print('after action ', next_state,reward)
        alpha = 1
        delta = reward + self.discount*self.grid.V[tuple(next_state)]-self.grid.V[tuple(current_state)]
#        print(self.grid.V[(current_state)])
        self.grid.V[tuple(current_state)] += alpha*delta
        
    def estimate(self,nb_iter):
        for _ in range(nb_iter):
            self.update()
        return self.grid.V

G = Ex2()
P = Policy(G.actions)
Test = TD0(G,P,.9)
print(Test.estimate(1000))
print(G.V[(0,1)])

    
    
    