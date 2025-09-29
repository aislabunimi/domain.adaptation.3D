#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from habitat_ros_bridge.msg import Sensors

def callback(msg):
    pub_rgb.publish(msg.rgb)
    pub_depth.publish(msg.depth)
    pub_sem.publish(msg.sem)

if __name__ == '__main__':
    rospy.init_node('Sensor_republisher')

    rospy.Subscriber("/habitat/scene/sensors", Sensors, callback)

    pub_rgb = rospy.Publisher("/sensors/image", Image, queue_size=1)
    pub_depth = rospy.Publisher("/sensors/depth", Image, queue_size=1)
    pub_sem = rospy.Publisher("/sensors/sem", Image, queue_size=1)

    rospy.spin()