#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:25:43 2017

@author: romain
"""
import numpy as np
import time
import matplotlib.pyplot as plt


class Gambler:
    def __init__(self,prob,expectation = 100):
        self.prob = prob
        self.expectation = expectation
        self.reset()
    
    def get_actions(self,state):
        if state not in ['win','lose']:
            return range(1,min(state,self.expectation-state)+1)
        return list()
    
    def transition(self,state,action):
        good = state+action
        if good >= self.expectation:
            good = 'win'
        bad = state - action
        if state - action <= 0:
            bad = 'lose'
        return {good:(self.prob,int(good=='win')),bad:(1-self.prob,0)}
    
    def _update(self,state,action,discount):
        transitions = self.transition(state,action)
        res = 0
        for nstates,truc in transitions.items():
            res += truc[0]*(truc[1]+discount*self.V[nstates])
        return res

    def update(self,state,discount):
        if state not in ['win','lose']:
            return (self._update(state,action,discount) for action in self.get_actions(state))
        return (0,0)
    def reset(self):
        self.V = {i:0 for i in range(1,self.expectation)}
        self.V['lose'] = 0
        self.V['win'] = 0
        self.trained = False
    
    def value_iteration(self,discount = 1,eps=.000001):
        self.reset()
        i = 0
        while True:
            i+=1
            delta = 0
            for state in self.V:
                old_v = self.V[state]
                self.V[state] = max(self.update(state,discount))
                delta += np.abs(old_v-self.V[state])
            if delta <eps:
                break
        print("nb iter = ",i)
        self.trained = True
        
    def get_policy(self,discount = 1):
        if not self.trained:
            raise ValueError('you must train before')
        self.policy = list()
        for state in range(1,self.expectation):
            self.policy.append(np.argmax(np.fromiter(self.update(state,discount),float))+1)
#            a = np.fromiter(self.update(state,discount),float)
#            print(np.where(self.update(state,discount) == max(self.update(state,discount))))
        return self.policy
    
    def see_policy(self,discount = 1):
        plt.scatter(np.arange(1,self.expectation),self.policy)
        plt.show()
            
            
g = Gambler(.4)
g.value_iteration()
#print(g.V)
g.get_policy()
#print(g.policy)
g.see_policy()
#print(g.policy)
#plt.plot([v for v in g.V.values() if v!=0])
#print([v for v in g.V.values() if v!=0])
plt.show()
            
                
        