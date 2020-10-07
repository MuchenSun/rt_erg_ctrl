import numpy as np
import matplotlib.pyplot as plt

import numba
from numba import jit
import time


grid = np.meshgrid(*[np.linspace(0, 4, 50) for _ in range(2)])
grid = np.stack((grid[0].reshape(-1), grid[1].reshape(-1)))
print(grid.shape)

@jit(nopython=True, parallel=True, cache=True)
def normal_pdf(mu, sig, x):
    return np.sqrt(np.linalg.det(2*np.pi*sig)) * np.exp(-0.5 * np.dot(np.dot(x-mu,np.linalg.inv(sig)),x-mu))

@jit(nopython=True, parallel=True, cache=True)
def fisher_info(mean, cov, grid):
    vals = []
    for i in range(len(grid)):
        vals.append(normal_pdf(mean, cov, grid[i]))
    return np.array(vals)

mean = np.array([1.3, 2.9])
cov = np.diag([0.8, 1.2])

for i in range(1000):
    start = time.time()
    vals = fisher_info(mean, cov, grid.T)
    print('elapsed time: {}'.format(time.time()-start))
print(vals)


