import numpy as np
from gym.spaces import Box
import sys
sys.path.append('../rt_erg_lib')
from barrier import Barrier


size = 4.0
explr_space = Box(np.array([0.0, 0.0]), np.array([size, size]), dtype=np.float32)

barrier = Barrier(explr_space)

loc = np.array([4.0, 2.0])
print('cost at {} is: {}'.format(loc, barrier.cost(loc)))

obstacles = [[1.5, 2.5]]
barrier.update_obstacles(obstacles)

loc = np.array([1.45, 2.47])
print('cost at {} is: {}'.format(loc, barrier.cost(loc)))

