import sys
sys.path.append('../rt_erg_lib')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from single_integrator_3d import SingleIntegrator
from ergodic_control import RTErgodicControl
from target_dist_3d import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi

env = SingleIntegrator()
model = SingleIntegrator()
t_dist = TargetDist(means=[0.2, 0.8, 3], num_nodes=2)
weights = {'R':np.diag([5 for _ in range(2)])}
erg_ctrl = RTErgodicControl(model, t_dist, horizon=15, num_basis=10, batch_size=-1, weights=weights)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

tf = 200
state = env.reset()
log = {'trajectory' : [state.copy()]}

for t in tqdm(range(tf)):
    ctrl = erg_ctrl(state, turtle_mode=False)
    state = env.step(ctrl)
    log['trajectory'].append(state)
print('doneee')
traj = np.array(log['trajectory'])

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')

xy, vals = t_dist.get_grid_spec()
ax.contourf(*xy, vals, levels=20)
traj_plot = ax.scatter([], [], s=10, c='r')
pointer = ax.plot([], [], color='orange')

def animate_at_t(t, traj_plot, pointer, traj_data, show_traj=True):
    if show_traj is True:
        traj_plot.set_offsets(traj_data[:t])
    else:
        traj_plot.set_offsets(traj_data[t])
    print(traj_data[t])
    pointer[0].set_xdata([traj_data[t][0], traj_data[t][0]+0.05])#*np.cos(traj_data[t][2])])
    pointer[0].set_ydata([traj_data[t][1], traj_data[t][1]+0.05])#*np.sin(traj_data[t][2])])
    return [traj_plot, pointer[0]]

anim = animation.FuncAnimation(fig, animate_at_t, frames=tf, interval=100, fargs=(traj_plot, pointer, traj, True), repeat=True)

plt.show()

"""
xy, vals = t_dist.get_grid_spec()
lamk = np.exp(-0.8*np.linalg.norm(erg_ctrl.basis.k, axis=1))

num_plts = 4
fig, axs = plt.subplots(2, num_plts, sharex=True, sharey=True)

for i in range(num_plts):
    tt = int(tf/num_plts*(i+1))
    axs[0,i].set_aspect('equal')
    axs[0,i].set_title('Time: {}'.format( tt ), fontsize=20)
    axs[0,i].contourf(*xy, vals, levels=20)
    xt = np.stack(log['trajectory'])
    axs[0,i].scatter(xt[:tt,0], xt[:tt,1], c='red', s=2)
    axs[0,i].axis('off')

    axs[1,i].set_aspect('equal')
    path = np.stack(log['trajectory'])[:tt,model.explr_idx]
    ck = convert_traj2ck(erg_ctrl.basis, path)
    val = convert_ck2dist(erg_ctrl.basis, ck)
    axs[1,i].contourf(*xy, val.reshape(50,50), levels=20)
    axs[1,i].axis('off')

plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.04, right=0.96)
# plt.tight_layout()
plt.show()
plt.close()

fdiffs = []
for i in tqdm(range(tf)):
    path = np.stack(log['trajectory'])[:i+1,model.explr_idx]
    ck = convert_traj2ck(erg_ctrl.basis, path)
    fdiff = lamk * (ck - erg_ctrl.phik)**2
    fdiff = np.sum(fdiff.reshape(-1,1))
    fdiffs.append(fdiff)
plt.plot(np.arange(tf), fdiffs)
plt.title('Receding-horizon ergodic cost', fontsize=20)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Ergodic Metric', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()
"""

