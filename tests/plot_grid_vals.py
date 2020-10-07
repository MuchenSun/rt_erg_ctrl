import numpy as np
import matplotlib.pyplot as plt


grid_vals = np.load('grid_vals.npy')
grid_vals = grid_vals.reshape(50, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
x, y = np.meshgrid(*[np.linspace(0, 4, 51) for _ in range(2)])
x = x[0:50, 0:50]
y = y[0:50, 0:50]

print(x.shape, y.shape, grid_vals.shape)

ax.contourf(x, y, grid_vals)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect('equal')

plt.show()

