#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:24:52 2017

@author: romain
"""

import numpy as np
import random as rd


class Env:
    def __init__(self):
        raise NotImplementedError('what the env looks like,\
                                  must implement the Q(s,a) representation')

    def step(self, action):
        raise NotImplementedError('return new_state and reward')

    def initQ(self):
        raise NotImplementedError

    def eps_greedy(self, epsilon, verbose=False):
        """ return an action using eps_greedy policy"""
        if rd.random() < 1-epsilon:
            greedy = np.argwhere(self.Q[self.state] == np.max(
                    self.Q[self.state]))
            return rd.choice(greedy.flatten())
        else:
            if verbose:
                print('pas greedy', self.state)
            return rd.randint(0, (self.Q[self.state].shape)[0]-1)

    def start(self):
        raise NotImplementedError

    def is_terminal(self):
        raise NotImplementedError

    def sarsa(self, nb_iter=10, alpha=.5, discount=1):
        self.initQ()
        a = 1
        for t in range(nb_iter):
            self.start()
            action = self.eps_greedy(.05)
            i = 0
            while True:
                a += 1
                i += 1
                old_state_x, old_state_y = self.state
                reward = self.step(action)
                if self.is_terminal():
                    self.Q[old_state_x, old_state_y, action] += (
                            alpha * (reward -
                                     self.Q[old_state_x, old_state_y, action]))
                    break
                else:
                    new_action = self.eps_greedy(.05)
                    self.Q[old_state_x, old_state_y, action] += (
                            alpha*(reward +
                                   discount*self.Q[self.state[0],
                                                   self.state[1], new_action] -
                                   self.Q[old_state_x, old_state_y, action]))
                action = new_action
            print('épisode ', t, 'finies en ', i, 'étapes')

    def Q_learning(self, nb_iter=10, alpha=.5, discount=1):
        self.initQ()
        for t in range(nb_iter):
            self.start()
            i = 0
            while True:
                action = self.eps_greedy(.05)
                old_state_x, old_state_y = self.state
                reward = self.step(action)
                if self.is_terminal():
                    self.Q[old_state_x, old_state_y, action] += (
                            alpha*(reward - self.Q[old_state_x,
                                                   old_state_y, action]))
                    break
                self.Q[old_state_x, old_state_y, action] += (
                        alpha*(reward +
                               discount*np.max(self.Q[self.state[0],
                                                      self.state[1]]) -
                               self.Q[old_state_x, old_state_y, action]))
                i += 1
            print('épisode ', t, 'finies en ', i, 'étapes')

    def get_policy(self):
        self.policy = np.argmax(self.Q, -1)


class Windy(Env):
    def __init__(self, long, haut, winds, begin, end):
        self.long = long
        self.haut = haut
        self.begin = begin
        self.winds = winds
        self.end = end

    def start(self):
        self.state = self.begin

    def initQ(self):
        self.Q = np.zeros((self.haut, self.long, 4))

    def is_terminal(self):
        return self.state == 'Terminal'

    def true_state_reward(self):
        y, x = self.state
        if x < 0:
            x = 0
        elif x >= self.long:
            x = self.long-1
        y = y - self.winds[x]
        if y < 0:
            y = 0
        elif y >= self.haut:
            y = self.haut - 1
        self.state = y, x
        return -1

    def step(self, action):
        """0:S 1:E 2:N 3:W"""
        if self.state == self.end:
            self.state = 'Terminal'
            return 1
        y, x = self.state
        if action == 0:
            self.state = y+1, x
        elif action == 1:
            self.state = y, x+1
        elif action == 2:
            self.state = y-1, x
        elif action == 3:
            self.state = y, x-1
        return self.true_state_reward()

    def get_policy(self):
        Env.get_policy(self)
        corresp = {0: 'S', 1: '->', 2: 'î', 3: '<-'}
        self.policy = [corresp[x] for x in self.policy.reshape(-1)]
        self.policy = np.array(self.policy).reshape(self.haut, -1)
        self.policy[self.begin] = "Start"
        self.policy[self.end] = "Goal"

    def get_episode(self, epsilon=.0001, verbose=False):
        self.start()
        while True:
            print(self.state)
            action = self.eps_greedy(epsilon)
            print(action)
            reward = self.step(action)
            print(reward)
            if self.state == 'Terminal':
                break
            print('-'*50)
        return None


class Cliff(Env):
    def __init__(self, long, haut, fall, begin, end):
        self.long = long
        self.haut = haut
        self.begin = begin
        self.end = end
        self.fall = fall

    def start(self):
        self.state = self.begin

    def initQ(self):
        self.Q = np.zeros((self.haut, self.long, 4))

    def is_terminal(self):
        return self.state == 'Terminal'

    def true_state_reward(self):
        y, x = self.state
        if (y == self.haut - 1) and (x in self.fall):
            self.start()
            return -100
        if y < 0:
            y = 0
        elif y == self.haut:
            y = self.haut - 1
        if x < 0:
            x = 0
        elif x == self.long:
            x = self.long - 1
        self.state = y, x
        return -1

    def step(self, action):
        """0:S 1:E 2:N 3:W"""
        if self.state == self.end:
            self.state = 'Terminal'
            return 1
        y, x = self.state
        if action == 0:
            self.state = y+1, x
        elif action == 1:
            self.state = y, x+1
        elif action == 2:
            self.state = y-1, x
        elif action == 3:
            self.state = y, x-1
        else:
            raise ValueError('lul')
        return self.true_state_reward()

    def get_policy(self):
        Env.get_policy(self)
        corresp = {0: 'V ', 1: '->', 2: 'A ', 3: '<-'}
#        self.policy = [[corresp[xx] for xx in x] for x in self.policy]
        self.policy = [corresp[x] for x in self.policy.reshape(-1)]
#        print(self.begin)
        self.policy = np.array(self.policy).reshape(self.haut, -1)
#        print(self.policy[10,3])
#        self.policy[self.begin] = "Start"
#        self.policy[self.end] = "Goal"


# =============================================================================
# C = Cliff(12, 4, np.arange(1, 11), (3, 0), (3, 11))
# C.Q_learning(1000)
# C.get_policy()
# print(C.policy)
# #W = Windy(10,7,[0,0,0,1,1,1,2,2,1,0],(3,0),(3,7))
# #W.sarsa(10)
# #W.get_policy()
# #print(W.policy)
# #W.Q_learning(10)
# #print(W.policy)
# =============================================================================


class Random_walk:
    def __init__(self):
        self.state = 2
        self.finished = False
        self.V = np.zeros(5)

    def step(self):
        move = 2*int(rd.random() < .5)-1
        self.state += move
        if self.state < 0:
            self.state = 'lose'
            return 0
        elif self.state > 4:
            self.state = 'win'
            return 1
        return 0

    def td_0(self, nb_iter=100, alpha=.1, discount=1):
        for _ in range(nb_iter):
            self.state = 3
            self.finished = False
            while True:
                old_state = self.state
                reward = self.step()
                if self.state in ['win', 'lose']:
                    self.V[old_state] += alpha*(reward-self.V[old_state])
                    break
                self.V[old_state] += alpha*(reward + (discount *
                                                      self.V[self.state] -
                                                      self.V[old_state]))


# =============================================================================
#
# #if __name__=='__main__':
# #    choix = input('Cliff or Windy\n')
# #    if 'c' in choix.lower():
# #        print('Cliff')
# #    elif 'w' in choix.lower():
# #        print('Windy')
# #    else:
# #        print('you suck')
#
# =============================================================================
