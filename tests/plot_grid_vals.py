import numpy as np
import matplotlib.pyplot as plt
import time


fig = plt.figure()
ax = fig.add_subplot(111)
x, y = np.meshgrid(*[np.linspace(0, 4, 51) for _ in range(2)])
x = x[1:51, 1:51]
y = y[1:51, 1:51]

while True:
    start_time = time.time()
    grid_vals = np.load('grid_vals.npy', allow_pickle=True)
    grid_vals = grid_vals.reshape(50, 50)

    ax.cla()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, grid_vals, levels=25)
    ax.set_xlim(0.08, 4)
    ax.set_ylim(0.08, 4)
    ax.set_aspect('equal')

    print('elapsed time: ', time.time()-start_time)

    plt.pause(0.9)

