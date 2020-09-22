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

        self.odom_sub = message_filters.Subscriber('/odom', Odometry)#, self.odom_callback)
        self.scan_sub = message_filters.Subscriber('/scan', LaserScan)#, self.scan_callback)
        self.obsv_pub = rospy.Publisher('/landmarks', Marker, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 1, 1)
        ts.registerCallback(self.callback)

        self.bearings = np.linspace(0, 2*pi, 360)
        self.start_time = time.time()
        print('initialization finished.')

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
        pass

    def loop(self):
        rospy.spin()


rospy.init_node('landmark_extraction_test_with_odom')
robot = TurtleBot()
robot.loop()


