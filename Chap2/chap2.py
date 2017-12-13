#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:16:43 2017

@author: romain
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k=10,initial = 0):
        self.k = k
        self.qstar = [rd.normalvariate(0,1) for _ in range(k)]
        self.bestAction = np.argmax(self.qstar)
        self.initial = initial
        self.q = initial*np.ones(k)
        self.n = np.zeros(k,dtype=int)
    
    def reset(self):
        self.q = self.initial*np.ones(self.k)
        self.n = np.zeros(self.k,dtype=int)
        
    def choose_action(self,epsilon):
        eps = rd.random()<epsilon
        if eps:
            return rd.randint(0,self.k-1)
        else:
            m = np.max(self.q)
            possible = [i for i,x in enumerate(self.q) if x == m]
            return rd.choice(possible)
    
    def get_reward(self,action):
        self.n[action] += 1
        reward = rd.normalvariate(self.qstar[action],1)
        self.q[action] += (reward-self.q[action])/self.n[action]
        return reward
    
    def compare(self,epsilon,nb_iter = 2000):
        res = 0
        for _ in range(nb_iter):
            action = self.choose_action(epsilon)
            res += self.get_reward(action)
        return res/nb_iter,self.n[self.bestAction]/nb_iter
    
    def average_reward(self,epsilon,nb_iter=500):
        res = list()
        for _ in range(nb_iter):
            self.reset()
            res.append(self.compare(epsilon)[0])
        return np.cumsum(res)/np.arange(1,len(res)+1)
    
    def good_bandit(self,epsilon,nb_iter=1000):
        res = list()
        for _ in range(nb_iter):
            self.reset()
            res.append(self.compare(epsilon)[1])
        return np.cumsum(res)/np.arange(1,len(res)+1)
        
    def __str__(self):
        return str(self.qstar)
    
    def __repr__(self):
        return str(self.qstar)

def diff_eps_accuracy():  
    b = Bandit()
    epsilon = [0,.01,.1]
    ax = plt.subplot(111)
    for i,eps in enumerate(epsilon):
        plt.plot(b.good_bandit(eps), label='epsilon = ' + str(eps))
    plt.ylim([0,1])
    plt.xlabel('runs')
    plt.ylabel('% accuracy')
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()

def diff_real_optim():
    bandit = [Bandit(10),Bandit(10,5)]
    q=['0','+5']
    ax = plt.subplot(111)
    for i,b in enumerate(bandit):
        plt.plot(b.good_bandit(.1), label='Q: ' + q[i])
    plt.ylim([0,1])
    plt.xlabel('runs')
    plt.ylabel('% accuracy')
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()

diff_eps_accuracy()
