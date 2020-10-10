import numpy as np
import copy


class Barrier(object):
    '''
    This class prevents the agent from
    going outside the exploration space
    '''
    def __init__(self, explr_space, pow=2, weight=100):
        self.explr_space = explr_space
        self.dl  = explr_space.high - explr_space.low
        self.pow = pow
        self.weight = weight
        self.eps = 0.01
        self.eps2 = 0.35#0.35
        self.pow2 = pow
        self.obstacles = []

    def update_obstacles(self, obstacles):
        self.obstacles = copy.copy(obstacles)

    def cost(self, x):
        '''
        Returns the actual cost of the barrier
        '''
        cost = 0.
        cost += np.sum((x > self.explr_space.high-self.eps) * (x - (self.explr_space.high-self.eps))**self.pow)
        cost += np.sum((x < self.explr_space.low+self.eps) * (x - (self.explr_space.low+self.eps))**self.pow)

        for obst in self.obstacles:
            cost += (np.sum((x-obst)**2) < self.eps2**2) * (np.sum((x-obst)**2) - self.eps2**2)**self.pow2
            # cost += (np.sqrt(np.sum((x-obst)**2)) < self.eps2) * (np.sqrt(np.sum((x-obst)**2)) - self.eps2)**self.pow2

        return self.weight * cost

    def dx(self, x):
        '''
        Returns the derivative of the barrier wrt to the exploration
        state
        '''
        dx = np.zeros(x.shape)
        dx += self.pow * (x > (self.explr_space.high-self.eps)) * (x - (self.explr_space.high-self.eps))**(self.pow-1)
        dx += self.pow * (x < (self.explr_space.low+self.eps)) * (x - (self.explr_space.low+self.eps))**(self.pow-1)

        for obst in self.obstacles:
            dx += self.pow2 * (np.sum((x-obst)**2) < self.eps2**2) * (np.sum((x-obst)**2) - self.eps2**2)**(self.pow2-1) * 2*(x-obst)
            # dx += self.pow2 * (np.sqrt(np.sum((x-obst)**2)) < self.eps2) * (np.sqrt(np.sum((x-obst)**2)) - self.eps2)**(self.pow2-1) * x/np.sqrt(np.sum(x-obst)**2)

        return self.weight * dx
