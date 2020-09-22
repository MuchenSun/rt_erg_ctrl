import rospy
import numpy as np

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan

from sklearn.cluster import Birch

pose = np.array([0.1, 0.1, 0.0])
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
        brc = Birch(n_clusters=None, threshold=0.05)
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
