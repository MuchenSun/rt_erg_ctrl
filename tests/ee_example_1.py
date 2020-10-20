import sys
sys.path.append('../rt_erg_lib')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from single_integrator import SingleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi

env = SingleIntegrator()
model = SingleIntegrator()
t_dist = TargetDist(num_nodes=2)
weights = {'R':np.diag([5 for _ in range(2)])}
erg_ctrl = RTErgodicControl(model, t_dist, horizon=15, num_basis=10, batch_size=-1, weights=weights)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

tf = 1200
log = {'trajectory' : []}
state = env.reset()
for t in tqdm(range(tf)):
    ctrl = erg_ctrl(state, turtle_mode=False)
    state = env.step(ctrl)
    log['trajectory'].append(state)
print('doneee')

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
