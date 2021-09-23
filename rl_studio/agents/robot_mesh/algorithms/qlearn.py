import gym
from gym import wrappers

import numpy as np
import random


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQValues(self, state, action):
        return self.q.get((state, action), 0.0)

    def selectAction(self, state, return_q=False):

        q = [self.getQValues(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            print("randomly choosing")
            return random.choice(list(self.actions))


        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2, done):
        maxqnew = max([self.getQValues(state2, a) for a in self.actions])

        if done == True:
            q_update = self.epsilon * (reward - self.getQValues(state1, action1))
        else:
            q_update = self.epsilon *(reward + self.gamma * maxqnew - self.getQValues(state1, action1))
        self.q[(state1, action1)] =self.getQValues(state1, action1) + q_update


    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
