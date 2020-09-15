"""
This version implemented a simple subscriber,
as a function, no class

Copied from ros_test_2, this version tries to
custimize space size
"""

import sys
sys.path.append('../rt_erg_lib')

###########################################
# basic function test

import numpy as np
import numpy.random as npr
from basis import Basis
from gym.spaces import Box

# configure size (assuming sqaure space)
size = 1.0

# define the exploration space as gym.Box
explr_space = Box(np.array([0.0, 0.0]), np.array([size, size]), dtype=np.float32)
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
print(barrier.cost(explr_space.sample()-size))
# test derivative of cost function wrt to pose
print(barrier.dx(explr_space.sample()-size))

###########################################
# target distribution test
from target_dist import TargetDist
from utils import convert_phi2phik, convert_phik2phi
import matplotlib.pyplot as plt

# define a target distribution object
t_dist = TargetDist()#TargetDist(size=size, means=[[1.0,1.0],[3.0,3.0]], cov=0.1)
# plot first fig, original target dist
fig1 = plt.figure()
ax1_1 = fig1.add_subplot(121)
ax1_1.set_aspect('equal')
xy, vals = t_dist.get_grid_spec()
ax1_1.contourf(*xy, vals, levels=20)
# compute phik from target distribution phi
# phik is determined once phi is determined
phik = convert_phi2phik(basis, t_dist.grid_vals, t_dist.grid, size)
print(phik.shape) # square of num_basis
# convert phik back to phi
phi = convert_phik2phi(basis, phik, t_dist.grid)
# plot the reconstructed phi
ax1_2 = fig1.add_subplot(122)
ax1_2.set_aspect('equal')
ax1_2.contourf(*xy, phi.reshape(50,50), levels=20)
plt.show()
plt.close()

############################################
# test ergodic controller
from turtlebot_dyn import TurtlebotDyn
from ergodic_control import RTErgodicControl
from utils import convert_ck2dist
from time import time

# configure simulation
env = TurtlebotDyn(size=size)
model = TurtlebotDyn(size=size)
weights = {'R': np.diag([10, 1])}
erg_ctrl = RTErgodicControl(model, t_dist, weights=weights, horizon=80, num_basis=5, batch_size=500)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)
tf = 1200

# ros configuration
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from numpy import sin, cos, pi, arcsin, arctan2

rospy.init_node('ergodic_controller')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
rate = rospy.Rate(10)

# start simulation
log = {'traj':[], 'ctrls':[], 'ctrl_seq':[], 'count':0}
state = env.reset(np.array([0.1, 0.1, 0.0]))

def callback(msg):
    # read odometry
    rx = msg.pose.pose.position.x
    ry = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    rth = arctan2(2*q.x*q.y-2*q.z*q.w, 1-2*q.y**2-2*q.z**2)
    rth = 2*pi - rth % (2*pi)
    state = np.array([rx, ry, rth]) #+ np.array([0.1, 0.1, 0.0])
    # compute control signal
    start_time = time()
    ctrl = erg_ctrl(state.copy())
    ctrl_lin = ctrl[0]
    ctrl_ang = ctrl[1]
    elapsed_time = time() - start_time
    print('[{}] elapsed time = {:4f}, pose = {}'.format(log['count'], elapsed_time, state))
    # send control signal
    vel_msg = Twist()
    vel_msg.linear.x = ctrl_lin
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0.0
    vel_msg.angular.x = 0.0
    vel_msg.angular.y = 0.0
    vel_msg.angular.z = ctrl_ang
    pub.publish(vel_msg)
    # log info
    log['count'] += 1
    log['traj'].append(state.copy())
    log['ctrls'].append(ctrl.copy())

sub = rospy.Subscriber('/odom', Odometry, callback)
rospy.spin()

tf = len(log['traj'])
traj = np.array(log['traj'])
ctrls = np.array(log['ctrls'])

# # randomly plot ctrl sequences
# fig3 = plt.figure()
# num_tests = 5
# for i in range(num_tests):
#     ax = fig3.add_subplot(num_tests, 1, i+1)
#     idx = np.random.randint(0, 80)
#     ax.plot(np.arange(80), ctrl_seq[idx])
# plt.show()
# plt.close()

# plot results
fig2 = plt.figure()
ax2_1 = fig2.add_subplot(311)
ax2_1.set_aspect('equal')
# ax2_1.set_xlim(0, 1)
# ax2_1.set_ylim(0, 1)
ax2_1.contourf(*xy, vals, levels=20)
ax2_1.scatter(traj[:,0], traj[:,1], c='r', s=1)
ax2_2 = fig2.add_subplot(312)
ax2_2.plot(np.arange(tf), ctrls[:,0], c='b')
ax2_2.axhline(+0.2, linestyle='--')
ax2_2.axhline(-0.2, linestyle='--')
ax2_3 = fig2.add_subplot(313)
ax2_3.plot(np.arange(tf), ctrls[:,1], c='y')
ax2_3.axhline(+2.8, linestyle='--')
ax2_3.axhline(-2.8, linestyle='--')
plt.show()

