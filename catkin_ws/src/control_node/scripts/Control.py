#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from ros_deeplabv3 import Finetune
import tf_conversions
import tf2_ros

from  geometry_msgs.msg import TransformStamped

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)
        
        # Initialize publishers
        self.init_publishers()
        
        # Initialize subscribers
        self.init_subscribers()
        
        # Other initialization
        self.running = True

    def broadcast_camera_pose(H, frames, stamp):

        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = frames[0]
        t.child_frame_id = frames[1]
        t.transform.translation.x = H[0, 3]
        t.transform.translation.y = H[1, 3]
        t.transform.translation.z = H[2, 3]
        q = tf_conversions.transformations.quaternion_from_matrix(H)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        br.sendTransform(t)

    def init_publishers(self):

        """Initialize all publishers"""
        #+Pose on func
        self.control_pub = rospy.Publisher('/sim/rgb', Image, queue_size=10)
        
        self.depth_pub = rospy.Publisher('/sim/depth', Image, queue_size=10)
        
        self.deeplab_pub = rospy.Publisher('/sim/mesh_map', PointCloud2, queue_size=10)

        self.semantic_pub= rospy.Publisher('/sim/semantic', Image, queue_size=10)

        self.finetune_pub= rospy.Publisher('/finetune_model', Finetune, queue_size=10)

        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map',Bool, queue_size=10 )
        
    def init_subscribers(self):
        """Initialize all subscribers"""

        # RGB+DEPTH(+Pose on func)
        rgb_topic = rospy.get_param('~rgb_topic', '/habitat/rgb')
        depth_topic = rospy.get_param('~depth_topic', '/habitat/depth')
        tf_topic= rospy.get_param('~tf_topic', '/habitat/tf')
        rospy.Subscriber(rgb_topic, Image, self.rgb_callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.Subscriber(tf_topic, TransformStamped, self.pose_callback)
        
        # From Kimera
        rospy.Subscriber('/kimera/mesh_map', OccupancyGrid, self.mesh_map_callback)
        rospy.Subscriber('/kimera/labels', Image, self.kimera_labels_callback)
        
        # From DeepLabV3
        rospy.Subscriber("/deeplab/segmented_image", Image, self.deeplab_labels_callback)
        
    def rgb_callback(self, msg):
        """Callback for RGB images from habitat"""
        # Process RGB image
        # Forward to DeepLabV3
        self.deeplab_pub.publish(msg)
        
    def depth_callback(self, msg):
        """Callback for depth images from habitat"""
        # Process depth image
        # Forward to Kimera
        self.kimera_pub.publish(msg)
        
    def pose_callback(self, msg):
        """Callback for pose information"""
        # Forward pose to exploration and Kimera
        self.exploration_pub.publish(msg)
        self.kimera_pub.publish(msg)
        
    def exploration_status_callback(self, msg):
        """Callback for exploration status"""
        # Handle exploration status updates
        pass
        
    def mesh_map_callback(self, msg):
        """Callback for mesh map from Kimera"""
        # Process mesh map
        pass
        
    def kimera_labels_callback(self, msg):
        """Callback for labels from Kimera"""
        # Process labels
        pass
        
    def deeplab_labels_callback(self, msg):
        """Callback for labels from DeepLab"""
        # Process labels
        # Could forward to finetuning module
        pass
        
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown() and self.running:
            # Main control logic here
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass