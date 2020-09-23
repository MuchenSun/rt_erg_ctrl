import numpy as np
from numpy import pi, sin, cos
from gym.spaces import Box

class TurtlebotDyn(object):

    def __init__(self, size=1.0, dt=0.1):
        self.observation_space = Box(np.array([0.0, 0.0, 0.0]),
                                     np.array([size, size, 2*pi]),
                                     dtype=np.float32)
        self.action_space = Box(np.array([-1.0, -1.0]),
                                np.array([+1.0, +1.0]),
                                dtype=np.float32)
        self.explr_space = Box(np.array([0.0, 0.0]),
                               np.array([size, size]),
                               dtype=np.float32)
        self.explr_idx = [0, 1]
        self.dt = dt#0.1

    def fdx(self, x, u):
        self.A =  np.array([[ 0.0, 0.0,-sin(x[2])*u[0]],
                            [ 0.0, 0.0, cos(x[2])*u[0]],
                            [ 0.0, 0.0,            0.0]])
        return self.A.copy()

    def fdu(self, x):
        self.B = np.array([[ cos(x[2]), 0.0],
                           [ sin(x[2]), 0.0],
                           [       0.0, 1.0]])
        return self.B.copy()

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(0.1, 0.9, size=3)
        else:
            self.state = state.copy()
        return self.state.copy()

    def f(self, x, u):
        return np.array([cos(x[2])*u[0], sin(x[2])*u[0], u[1]])

    def step(self, a):
        self.state = self.state + self.f(self.state, a) * self.dt
        self.state[2] = self.state[2] % (2*pi)
        return self.state.copy()


