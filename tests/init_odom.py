import rospy
from nav_msgs.msg import Odometry
import copy


pub = rospy.Publisher('/new_odom', Odometry, queue_size=10)

def callback(msg):
    new_msg = copy.deepcopy(msg)
    new_msg.pose.pose.position.x += 0.1
    new_msg.pose.pose.position.y += 0.1
    pub.publish(new_msg)

rospy.init_node('init_odom')
sub = rospy.Subscriber('/odom', Odometry, callback)
rospy.spin()


