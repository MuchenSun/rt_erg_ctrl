import rospy
import numpy as np
from numpy import pi, cos, sin, arcsin, arctan2
import time

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
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

        self.odom_sub = message_filters.Subscriber('/odom', Odometry)#, self.odom_callback)
        self.scan_sub = message_filters.Subscriber('/scan', LaserScan)#, self.scan_callback)
        self.obsv_pub = rospy.Publisher('/landmarks', Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10, 0.01)
        ts.registerCallback(self.callback)

        self.bearings = np.linspace(0, 2*pi, 360)
        self.start_time = time.time()

        # ergodic control config
        self.size = 1.0
        self.explr_space = Box(np.array([0.0,0.0]), np.array([self.size,self.size]), dtype=np.float64)
        self.basis = Basis(explr_space=self.explr_space, num_basis=5)
        self.t_dist = TargetDist()
        self.phik = convert_phi2phik(self.basis, self.t_dist.grid_vals, self.t_dist.grid, self.size)
        self.weights = {'R': np.diag([10, 1])}
        self.model = TurtlebotDyn(size=self.size, dt=0.2)
        self.erg_ctrl = RTErgodicControl(self.model, self.t_dist, weights=self.weights, horizon=80, num_basis=5, batch_size=500)
        self.erg_ctrl.phik = self.phik
        self.log = {'traj':[], 'ctrls':[], 'ctrl_seq':[], 'count':0}

    def callback(self, odom_msg, scan_msg):
        print('-----------------------------------------')
        # process odometry message
        rx = odom_msg.pose.pose.position.x
        ry = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        rth = arctan2(2*q.x*q.y-2*q.z*q.w, 1-2*q.y**2-2*q.z**2)
        rth = 2*pi - rth % (2*pi)
        pose = np.array([rx, ry, rth])
        self.pose = pose.copy()

        # process scan message
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
                if fit_cov < 0.001 and seg.shape[0]>=5:
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

        # send control
        ctrl = self.erg_ctrl(pose.copy())
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
ax2_1.contourf(*xy, vals, levels=20)
traj = np.array(robot.log['traj'])
ctrls = np.array(robot.log['ctrls'])
tf = robot.log['count']
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
