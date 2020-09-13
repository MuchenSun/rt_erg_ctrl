import sys
sys.path.append('../rt_erg_lib')

###########################################
# basic function test

import numpy as np
import numpy.random as npr
from basis import Basis
from gym.spaces import Box

# define the exploration space as gym.Box
explr_space = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]), dtype=np.float32)
# define the basis object
basis = Basis(explr_space=explr_space, num_basis=5)
# simulate/randomize a trajectory
xt = [explr_space.sample() for _ in range(10)]
# print indices for all basis functions
print('indices for all basis functions: ')
print(basis.k) # amount is square of num_basis
# test basis function, the input is a pose
print(basis.fk(xt[0]))
# test derivative of basis function wrt a pose
print(basis.dfk(xt[0]))
# hk, even computed in the source code, is not
# used in the end, so we temporarily ignore it

###########################################
# compute trajectory to ck using basis function
from utils import convert_traj2ck
ck = convert_traj2ck(basis, xt)
print('ck: ')
print(ck)

###########################################
# barrier function test
from barrier import Barrier

# define the Barrier object
barrier = Barrier(explr_space)
# test cost function
print(barrier.cost(explr_space.sample()-1.0))
# test derivative of cost function wrt to pose
print(barrier.dx(explr_space.sample()-1.0))

###########################################
# target distribution test
from target_dist import TargetDist
from utils import convert_phi2phik, convert_phik2phi
import matplotlib.pyplot as plt

# define a target distribution object
t_dist = TargetDist()
# plot first fig, original target dist
fig1 = plt.figure()
ax1_1 = fig1.add_subplot(121)
ax1_1.set_aspect('equal')
ax1_1.set_xlim(0, 1)
ax1_1.set_ylim(0, 1)
xy, vals = t_dist.get_grid_spec()
ax1_1.contourf(*xy, vals, levels=20)
# compute phik from target distribution phi
# phik is determined once phi is determined
phik = convert_phi2phik(basis, t_dist.grid_vals, t_dist.grid)
print(phik.shape) # square of num_basis
# convert phik back to phi
phi = convert_phik2phi(basis, phik, t_dist.grid)
# plot the reconstructed phi
ax1_2 = fig1.add_subplot(122)
ax1_2.set_aspect('equal')
ax1_2.set_xlim(0, 1)
ax1_2.set_ylim(0, 1)
ax1_2.contourf(*xy, phi.reshape(50,50))
# plt.pause(0.1)
plt.close()

############################################
# test ergodic controller
from single_integrator import SingleIntegrator
from ergodic_control import RTErgodicControl
from utils import convert_ck2dist

# configure simulation
env = SingleIntegrator()
model = SingleIntegrator()
weights = {'R': np.eye(model.action_space.shape[0]) * 5}
erg_ctrl = RTErgodicControl(model, t_dist, weights=weights, horizon=15, \
                                           num_basis=5, batch_size=100)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)
tf = 600

# start simulation
log = {'traj':[], 'ctrls':[]}
state = env.reset()
for t in range(tf):
    log['traj'].append(state)
    ctrl = erg_ctrl(state)
    log['ctrls'].append(ctrl)
    state = env.step(ctrl)
print('simulation finished :)')
traj = np.array(log['traj'])
ctrls = np.array(log['ctrls'])

# plot results
taxis = np.arange(tf)
fig2 = plt.figure()
ax2_1 = fig2.add_subplot(211)
ax2_1.set_aspect('equal')
ax2_1.set_xlim(0, 1)
ax2_1.set_ylim(0, 1)
ax2_1.contourf(*xy, vals, levels=20)
ax2_1.scatter(traj[:,0], traj[:,1], c='r', s=1)
ax2_2 = fig2.add_subplot(212)
ax2_2.plot(np.arange(tf), ctrls)
plt.show()


