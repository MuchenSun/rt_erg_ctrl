import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


rospy.init_node('rviz_marker_test')
#pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=1)
pub = rospy.Publisher('/landmarks', Marker, queue_size=1)

landmarks = np.random.uniform(0, 4, size=(5,2))
print(landmarks)

line_list = Marker()
line_list.header.frame_id = 'odom'
line_list.header.stamp = rospy.Time.now()
line_list.ns = 'landmark_point'
line_list.action = Marker.ADD
line_list.pose.orientation.w = 1.0
line_list.id = 0
line_list.type = Marker.CUBE_LIST#LINE_LIST

line_list.scale.x = 0.05
line_list.scale.y = 0.05
line_list.scale.z = 0.5
line_list.color.b = 1.0
line_list.color.a = 1.0

point_list = []
for landmark in landmarks:
    '''
    p1 = Point()
    p1.x = landmark[0]
    p1.y = landmark[1]
    p1.z = 0.0
    '''

    p2 = Point()
    p2.x = landmark[0]
    p2.y = landmark[1]
    p2.z = 0.25

    # line_list.points.append(p1)
    line_list.points.append(p2)

while not rospy.is_shutdown():
    pub.publish(line_list)
