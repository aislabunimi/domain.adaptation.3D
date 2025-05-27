#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from kimera_interfacer.msg import SyncSemantic

def callback(msg):
    pub_rgb.publish(msg.image)
    pub_depth.publish(msg.depth)
    pub_sem.publish(msg.sem)

if __name__ == '__main__':
    rospy.init_node('syncsemantic_republisher')

    rospy.Subscriber("/sync_semantic", SyncSemantic, callback)

    pub_rgb = rospy.Publisher("/sync_semantic/image", Image, queue_size=1)
    pub_depth = rospy.Publisher("/sync_semantic/depth", Image, queue_size=1)
    pub_sem = rospy.Publisher("/sync_semantic/sem", Image, queue_size=1)

    rospy.spin()