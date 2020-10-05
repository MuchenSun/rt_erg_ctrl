"""
This version is based on ergodic_slam_2, it's used
to test obstacle collision avoidance with barrier functions
"""

import rospy
import numpy as np
from numpy import pi, cos, sin, arcsin, arctan2
from scipy import linalg
import time
import math

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
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

from copy import copy


class TurtleBot(object):

    def __init__(self):
        # basic config
        self.pose = np.array([0.1, 0.1, 0.])
        self.obsv = []

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

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ctrl_sub = rospy.Subscriber('/ctrl_flag', Bool, self.ctrl_callback)
        self.obsv_pub = rospy.Publisher('/landmarks', Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_pub = rospy.Publisher('/test_path', Path, queue_size=1)
        self.map_pub = rospy.Publisher('/landmarks_map', Marker, queue_size=1)
        self.ekf_pub = rospy.Publisher('/ekf_odom', Odometry, queue_size=1)

        self.path_msg = Path()
        self.odom_header = None

        #######
        # for test only
        self.old_obsv = []
        self.curr_obsv = []
        self.lm_table = []

        #######
        # ekf
        self.ekf_mean = np.array([0.0, 0.0, 0.0])
        self.ekf_cov = np.diag([1e-09 for _ in range(3)])
        self.ekf_R = np.diag([0.02, 0.02, 0.01])
        self.ekf_Q = np.diag([0.03, 0.01])
        self.init_flag = False

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
        print('odom child_frame_id: {}'.format(odom_msg.child_frame_id))

        '''
        self.path_msg.header = copy(odom_msg.header)
        curr_pose = copy(odom_msg.pose.pose)
        curr_pose_stamped = PoseStamped()
        curr_pose_stamped.header = copy(odom_msg.header)
        curr_pose_stamped.pose = curr_pose
        self.path_msg.poses.append(copy(curr_pose_stamped))
        '''
        self.odom_header = copy(odom_msg.header)
        if self.init_flag is False:
            self.ekf_mean[0] = rx
            self.ekf_mean[1] = ry
            self.ekf_mean[2] = rth
            self.init_flag = True

    def scan_callback(self, scan_msg):
        print('\n\n-------------------scan callback----------------------')
        start_time = time.time()

        # process scan message
        # pose = self.pose.copy()
        pose = self.ekf_mean[0:3].copy()
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
        original_range = ranges[dist_flag]
        original_bearing = bearings[dist_flag]
        print('original_range.shape: {}'.format(original_range.shape))

        self.obsv = []
        self.raw_scan = []
        if len(points) > 0:
            brc = Birch(n_clusters=None, threshold=0.05)
            brc.fit(points)
            labels = brc.predict(points)
            u_labels = np.unique(labels)
            for l in u_labels:
                seg_idx = np.where(labels==l)
                seg = points[seg_idx]
                raw_scan = np.array([original_range[seg_idx], original_bearing[seg_idx]]).T
                if seg.shape[0] <= 1:
                    fit_cov = 10
                else:
                    fit_cov = np.trace(np.cov(seg.T))
                lm = seg.mean(axis=0)
                if fit_cov < 0.001 and seg.shape[0]>=5 and lm[0]>0 and lm[0]<4 and lm[1]>0 and lm[1]<4:
                    self.obsv.append(lm.copy())
                    self.raw_scan.append(raw_scan.mean(axis=0))

        print('odom: {}\nlandmarks:\n{}'.format(pose, len(self.obsv)))

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
        print('----ctrl callback----')
        pose = self.pose.copy()
        pose[2] = self.normalize(pose[2])
        self.old_obsv = np.array(self.curr_obsv, copy=True)
        self.curr_obsv = np.array(self.obsv, copy=True)
        update_flag =np.array_equal(self.old_obsv, self.curr_obsv)

        ########
        # landmarks table test
        ########
        num_lm = int(0.5 * (self.ekf_mean.shape[0] - 3))
        self.obsv_table = []
        tid = 0
        for olm in self.curr_obsv:
            # for each observation
            '''
            tflag = 1
            for lid in range(num_lm):
                # compare it with observed landmark
                tlm = self.ekf_mean[3+lid*2:5+lid*2]
                lm_diff = np.sum((olm-tlm)**2)
                if lm_diff < 0.3:
                    # this is observed landmark
                    # self.ekf_mean[3+lid*2] = olm[0]
                    # self.ekf_mean[4+lid*2] = olm[1]
                    self.obsv_table.append(lid)
                    tflag = 0
                    break
                else:
                    pass
            '''
            ds_score = []
            for lid in range(num_lm):
                # compare it with observed landmarks
                tlm = self.ekf_mean[3+lid*2 : 5+lid*2]
                lm_diff = np.sum((olm-tlm)**2)
                ds_score.append(lm_diff)

            if len(ds_score) > 0:
                if min(ds_score) > 0.5: # new landmark
                    self.obsv_table.append(num_lm+tid)
                    self.ekf_mean = np.concatenate((self.ekf_mean, olm))
                    self.ekf_cov = np.block([[self.ekf_cov, np.zeros((self.ekf_cov.shape[0],2))],
                                             [np.zeros((2,self.ekf_cov.shape[0])), np.eye(2)*1000]])
                    tid += 1
                else:
                    lidx = np.argmin(ds_score)
                    self.obsv_table.append(lidx)
            else:
                self.obsv_table.append(num_lm+tid)
                self.ekf_mean = np.concatenate((self.ekf_mean, olm))
                self.ekf_cov = np.block([[self.ekf_cov, np.zeros((self.ekf_cov.shape[0],2))],
                                         [np.zeros((2,self.ekf_cov.shape[0])), np.eye(2)*1000]])
                tid += 1

        print('obsv: {}'.format(self.curr_obsv))
        print('obsv table: {}'.format(self.obsv_table))

        cube_list = Marker()
        cube_list.header.frame_id = 'odom'
        cube_list.header.stamp = rospy.Time.now()
        cube_list.ns = 'landmark_map'
        cube_list.action = Marker.ADD
        cube_list.pose.orientation.w = 1.0
        cube_list.id = 0
        cube_list.type = Marker.CUBE_LIST

        cube_list.scale.x = 0.05
        cube_list.scale.y = 0.05
        cube_list.scale.z = 0.5
        cube_list.color.r = 1.0
        cube_list.color.g = 1.0
        cube_list.color.a = 1.0

        for i in range(num_lm+tid):
            landmark = self.ekf_mean[3+i*2:5+i*2]
            p = Point()
            p.x = landmark[0]
            p.y = landmark[1]
            p.z = 0.25
            cube_list.points.append(p)

        self.map_pub.publish(cube_list)

        ########
        # ekf
        ########
        prev_ctrl = np.array([0., 0.])
        if len(self.log['ctrls']) > 0:
            prev_ctrl = self.log['ctrls'][-1].copy()
        G = np.eye(self.ekf_mean.shape[0])
        G[0][2] = -np.sin(self.ekf_mean[2]) * prev_ctrl[0] * 0.1
        G[1][2] =  np.cos(self.ekf_mean[2]) * prev_ctrl[1] * 0.1
        num_lm = int(0.5 * (self.ekf_mean.shape[0]-3))
        BigR = np.block([
                    [self.ekf_R, np.zeros((3, 2*num_lm))],
                    [np.zeros((2*num_lm, 3)), np.zeros((2*num_lm, 2*num_lm))]
                ])
        self.ekf_cov = G @ self.ekf_cov @ G.T + BigR
        self.ekf_mean[0] = pose[0]
        self.ekf_mean[1] = pose[1]
        self.ekf_mean[2] = pose[2]
        self.ekf_mean[2] = self.normalize(self.ekf_mean[2])
        # self.ekf_mean[0] += np.cos(self.ekf_mean[2]) * prev_ctrl[0] * 0.1
        # self.ekf_mean[1] += np.sin(self.ekf_mean[2]) * prev_ctrl[0] * 0.1
        # self.ekf_mean[2] += prev_ctrl[1] * 0.1

        # if update_flag is False:
        if True:
            num_obsv = len(self.curr_obsv)
            H = np.zeros((2*num_obsv, self.ekf_mean.shape[0]))
            r = self.ekf_mean[0:3].copy()
            ref_obsv = []

            for i in range(num_obsv):
                idx = i*2
                lid = self.obsv_table[i]
                lm = self.ekf_mean[3+lid*2 : 5+lid*2]
                zr = np.sqrt((r[0]-lm[0])**2 + (r[1]-lm[1])**2)

                H[idx][0] = (r[0]-lm[0]) / zr
                H[idx][1] = (r[1]-lm[1]) / zr
                H[idx][2] = 0
                H[idx][3+2*lid] = -(r[0]-lm[0]) / zr
                H[idx][4+2*lid] = -(r[1]-lm[1]) / zr

                H[idx+1][0] = -(r[1]-lm[1]) / zr**2
                H[idx+1][1] =  (r[0]-lm[0]) / zr**2
                H[idx+1][2] = -1
                H[idx+1][3+2*lid] =  (r[1]-lm[1]) / zr**2
                H[idx+1][4+2*lid] = -(r[0]-lm[0]) / zr**2

                ref_obsv.append(self.range_bearing(r, lm))


            ref_obsv = np.array(ref_obsv)
            BigQ = linalg.block_diag(*[self.ekf_Q for _ in range(num_obsv)])

            mat1 = np.dot(self.ekf_cov, H.T)
            mat2 = np.dot(np.dot(H, self.ekf_cov), H.T)
            mat3 = np.linalg.inv(mat2 + BigQ)
            K = np.dot(mat1, mat3)

            ori_obsv = []
            for obsv in self.curr_obsv:
                temp = self.range_bearing(r, obsv)
                temp[1] = self.normalize(temp[1])
                ori_obsv.append(temp.copy())
            ori_obsv = np.array(ori_obsv)
            # raw_scan = np.array(self.raw_scan)
            if len(ori_obsv) == 0:
                pass
            else:
                # raw_scan[:,1] = self.normalize(raw_scan[:,1])
                delta_z = ori_obsv - ref_obsv
                delta_z[:,1] = self.normalize(delta_z[:,1])
                delta_z = delta_z.reshape(-1)
                self.ekf_mean += K @ delta_z
                self.ekf_cov -= K @ H @ self.ekf_cov

        print('r: {} | {}'.format(pose, self.ekf_mean[0:3]))
        print(len(self.ekf_mean))
        ekf_odom = Odometry()
        ekf_odom.header = copy(self.odom_header)
        ekf_odom.child_frame_id = "base_footprint"
        ekf_odom.pose.pose.position.x = self.ekf_mean[0]
        ekf_odom.pose.pose.position.y = self.ekf_mean[1]
        ekf_odom.pose.covariance[0] = self.ekf_cov[0][0]
        ekf_odom.pose.covariance[1] = self.ekf_cov[1][1]
        self.ekf_pub.publish(ekf_odom)
        print('ekf cov: ', np.trace(self.ekf_cov))

        ########
        # ctrl
        ########
        idx = self.log['count'] % 10
        self.erg_ctrl.barr.update_obstacles(self.obsv)
        # _, ctrl_seq = self.erg_ctrl(pose.copy(), seq=True)
        _, ctrl_seq = self.erg_ctrl(self.ekf_mean[0:3].copy(), seq=True)

        if idx == 0:
            print('update ctrl seq')
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
            print('follow ctrl seq')
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

        # publish predicted trajectory
        self.path_msg = Path()
        self.path_msg.header = copy(self.odom_header)
        dummy_pose = pose.copy()
        for i in range(idx, 80):
            dummy_ctrl = self.ctrl_seq[i]
            dummy_pose += 0.1 * np.array([cos(dummy_pose[2])*dummy_ctrl[0],
                                          sin(dummy_pose[2])*dummy_ctrl[0],
                                          dummy_ctrl[1]])
            pose_msg = PoseStamped()
            pose_msg.header = copy(self.odom_header)
            pose_msg.pose.position.x = dummy_pose[0]
            pose_msg.pose.position.y = dummy_pose[1]
            self.path_msg.poses.append(copy(pose_msg))
        self.path_pub.publish(self.path_msg)


    def loop(self):
        rospy.spin()

    def normalize(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def range_bearing(self, agent, landmark):
        delta = landmark - agent[0:2]
        rangee = np.sqrt(np.dot(delta.T, delta))
        bearing = math.atan2(delta[1], delta[0]) - agent[2]
        bearing = self.normalize(bearing)
        return np.array([rangee, bearing])

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
# plt.show()
