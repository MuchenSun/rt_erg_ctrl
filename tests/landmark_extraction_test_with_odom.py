import rospy
import numpy as np
from numpy import pi, cos, sin, arcsin, arctan2
import time

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import message_filters

from sklearn.cluster import Birch


class TurtleBot(object):

    def __init__(self):
        self.pose = np.array([0.1, 0.1, 0.])
        self.obsv = []

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.bearings = np.linspace(0, 2*pi, 360)
        self.start_time = time.time()

    def odom_callback(self, msg):
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        rth = arctan2(2*q.x*q.y-2*q.z*q.w, 1-2*q.y**2-2*q.z**2)
        rth = 2*pi - rth % (2*pi)
        self.pose = np.array([rx, ry, rth])
        print(self.pose)

    def scan_callback(self, msg):
        pose = self.pose.copy()
        bearings = self.bearings.copy()

        ranges = np.array(msg.ranges)
        inf_flag = (-1 * np.isinf(ranges).astype(int) + 1)
        ranges = np.nan_to_num(ranges) * inf_flag

        euc_coord_x = pose[0] + np.cos(bearings-pose[2]) * ranges
        euc_coord_y = pose[1] + np.sin(bearings-pose[2]) * ranges
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
                if fit_cov < 0.001 and seg.shape[0]>=3:
                    self.obsv.append(seg.mean(axis=0))
            print(self.obsv)

    def loop(self):
        rospy.spin()


rospy.init_node('landmark_extraction_test_with_odom')
robot = TurtleBot()
robot.loop()


"""

pose = np.array([0.0, 0.0, 0.0])
bearings = np.linspace(0, 2*np.pi, 360)
pub = rospy.Publisher('/landmarks', Marker, queue_size=1)

def callback(msg):
    ranges = np.array(msg.ranges)
    inf_flag = (-1 * np.isinf(ranges).astype(int) + 1)
    ranges = np.nan_to_num(ranges) * inf_flag

    euc_coord_x = pose[0] + np.cos(bearings-pose[2]) * ranges
    euc_coord_y = pose[1] + np.sin(bearings-pose[2]) * ranges
    dist_flag = np.where( (euc_coord_x-pose[0])**2 + \
                    (euc_coord_y-pose[1])**2 != 0.0)[0]
    points = np.array([euc_coord_x, euc_coord_y]).T
    points = points[dist_flag]

    if len(points) > 0:
        brc = Birch(n_clusters=None, threshold=0.01)
        brc.fit(points)
        labels = brc.predict(points)
        u_labels = np.unique(labels)
        landmarks = []
        for l in u_labels:
            seg_idx = np.where(labels==l)
            seg = points[seg_idx]
            if seg.shape[0] <= 1:
                fit_cov = 10
            else:
                fit_cov = np.trace(np.cov(seg.T))
            if fit_cov < 0.001 and seg.shape[0]>=3:
                landmarks.append(seg.mean(axis=0))

        if len(landmarks) > 0:
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

            for landmark in landmarks:
                p = Point()
                p.x = landmark[0]
                p.y = landmark[1]
                p.z = 0.25
                cube_list.points.append(p)
            pub.publish(cube_list)

rospy.init_node('landmark_extraction_test')
sub = rospy.Subscriber('/scan', LaserScan, callback)
rospy.spin()

"""
