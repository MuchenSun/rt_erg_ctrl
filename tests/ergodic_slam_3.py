"""
This version is based on ergodic_slam_2, it's used
to test obstacle collision avoidance with barrier functions
"""

import rospy
import numpy as np
from numpy import pi, cos, sin, arcsin, arctan2
import time

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import message_filters

from sklearn.cluster import Birch

import sys
sys.path.append('../rt_erg_lib')
from basis import Basis
from gym.spaces import Box
from target_dist import TargetDist
from utils import *
from ergodic_control import RTErgodicControl
from turtlebot_dyn import TurtlebotDyn


class TurtleBot(object):

    def __init__(self):
        # basic config
        self.pose = np.array([0.1, 0.1, 0.])
        self.obsv = []

        # ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.01)
        # ts.registerCallback(self.callback)

        self.bearings = np.linspace(0, 2*pi, 360)
        self.start_time = time.time()

        # ergodic control config
        self.size = 4.0
        self.explr_space = Box(np.array([0.0,0.0]), np.array([self.size,self.size]), dtype=np.float64)
        self.basis = Basis(explr_space=self.explr_space, num_basis=10)
        # self.t_dist = TargetDist()
        self.t_dist = TargetDist(means=[[1.0,1.0],[3.0,3.0]], cov=0.1, size=self.size)
        self.phik = convert_phi2phik(self.basis, self.t_dist.grid_vals, self.t_dist.grid, self.size)
        self.weights = {'R': np.diag([10, 1])}
        self.model = TurtlebotDyn(size=self.size, dt=0.1)
        self.erg_ctrl = RTErgodicControl(self.model, self.t_dist, weights=self.weights, horizon=80, num_basis=10, batch_size=500)

        self.obstacles = np.array([[1.,2.], [2.,1.], [2.,2.], [3.,1.], [1.,3.], [2.,3.], [3.,2.]])
        self.erg_ctrl.barr.update_obstacles(self.obstacles)

        self.erg_ctrl.phik = self.phik
        self.log = {'traj':[], 'ctrls':[], 'ctrl_seq':[], 'count':0}

        # self.odom_sub = message_filters.Subscriber('/odom', Odometry)#, self.odom_callback)
        # self.scan_sub = message_filters.Subscriber('/scan', LaserScan)#, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ctrl_sub = rospy.Subscriber('/ctrl_flag', Bool, self.ctrl_callback)
        self.obsv_pub = rospy.Publisher('/landmarks', Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def odom_callback(self, odom_msg):
        rx = odom_msg.pose.pose.position.x
        ry = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        rth = arctan2(2*q.x*q.y-2*q.z*q.w, 1-2*q.y**2-2*q.z**2)
        rth = 2*pi - rth % (2*pi)
        # pose = np.array([rx, ry, rth])
        # self.pose = pose.copy()
        self.pose[0] = rx
        self.pose[1] = ry
        self.pose[2] = rth

    def scan_callback(self, scan_msg):
        print('-----------------------------------------')
        start_time = time.time()

        # process scan message
        pose = self.pose.copy()
        bearings = self.bearings.copy()

        ranges = np.array(scan_msg.ranges)
        inf_flag = (-1 * np.isinf(ranges).astype(int) + 1)
        ranges = np.nan_to_num(ranges) * inf_flag

        euc_coord_x = pose[0] + np.cos(bearings + pose[2]) * ranges
        euc_coord_y = pose[1] + np.sin(bearings + pose[2]) * ranges
        dist_flag = np.where( (euc_coord_x-pose[0])**2 + \
                        (euc_coord_y-pose[1])**2 != 0.0)[0]
        points = np.array([euc_coord_x, euc_coord_y]).T
        points = points[dist_flag]

        self.obsv = []
        if len(points) > 0:
            brc = Birch(n_clusters=None, threshold=0.05)
            brc.fit(points)
            labels = brc.predict(points)
            u_labels = np.unique(labels)
            for l in u_labels:
                seg_idx = np.where(labels==l)
                seg = points[seg_idx]
                if seg.shape[0] <= 1:
                    fit_cov = 10
                else:
                    fit_cov = np.trace(np.cov(seg.T))
                if fit_cov < 0.001 and seg.shape[0]>=4:
                    self.obsv.append(seg.mean(axis=0))

        print('odom: {}\nlandmarks:\n{}'.format(pose, self.obsv))

        # publish observed landmarks
        cube_list = Marker()
        cube_list.header.frame_id = 'odom'
        cube_list.header.stamp = rospy.Time.now()
        cube_list.ns = 'landmark_point'
        cube_list.action = Marker.ADD
        cube_list.pose.orientation.w = 1.0
        cube_list.id = 0
        cube_list.type = Marker.CUBE_LIST

        cube_list.scale.x = 0.05
        cube_list.scale.y = 0.05
        cube_list.scale.z = 0.5
        cube_list.color.b = 1.0
        cube_list.color.a = 1.0

        for landmark in self.obsv:
            p = Point()
            p.x = landmark[0]
            p.y = landmark[1]
            p.z = 0.25
            cube_list.points.append(p)

        self.obsv_pub.publish(cube_list)

        print('elasped time: {}'.format(time.time()-start_time))

    def ctrl_callback(self, ctrl_flag_msg):
        idx = self.log['count'] % 10 == 0
        pose = self.pose.copy()

        self.erg_ctrl.barr.update_obstacles(self.obsv)
        _, ctrl_seq = self.erg_ctrl(pose.copy(), seq=True)

        if idx:
            self.ctrl_seq = ctrl_seq.copy()

            ctrl = self.ctrl_seq[idx]
            ctrl_lin = ctrl[0]
            ctrl_ang = ctrl[1]
            vel_msg = Twist()
            vel_msg.linear.x = ctrl_lin
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = ctrl_ang
            self.ctrl_pub.publish(vel_msg)
        else:
            ctrl = self.ctrl_seq[idx]
            ctrl_lin = ctrl[0]
            ctrl_ang = ctrl[1]
            vel_msg = Twist()
            vel_msg.linear.x = ctrl_lin
            vel_msg.linear.y = 0.0
            vel_msg.linear.z = 0.0
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = ctrl_ang
            self.ctrl_pub.publish(vel_msg)

        # log
        self.log['count'] += 1
        self.log['traj'].append(pose.copy())
        self.log['ctrls'].append(ctrl.copy())

    def loop(self):
        rospy.spin()


rospy.init_node('landmark_extraction_test_with_odom')
robot = TurtleBot()
robot.loop()

# plot results
import matplotlib.pyplot as plt
fig2 = plt.figure()
ax2_1 = fig2.add_subplot(311)
ax2_1.set_aspect('equal')
# ax2_1.set_xlim(0, 1)
# ax2_1.set_ylim(0, 1)
xy, vals = robot.t_dist.get_grid_spec()
# ax2_1.contourf(*xy, vals, levels=20)
traj = np.array(robot.log['traj'])
ctrls = np.array(robot.log['ctrls'])
tf = robot.log['count']

path = np.stack(traj)[:,0:2]
ck = convert_traj2ck(robot.erg_ctrl.basis, path)
vals = convert_ck2dist(robot.erg_ctrl.basis, ck, size=robot.size)
vals = vals.reshape(50, 50)
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
