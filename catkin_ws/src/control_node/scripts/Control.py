#!/usr/bin/env python3
import rospy
import std_msgs
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from ros_deeplabv3 import Finetune
import tf_conversions
import tf2_ros
from kimera_interfacer.msg import SyncSemantic
from habitat_ros_bridge.msg import Sensors


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


    def init_publishers(self):

        """Initialize all publishers"""
        #+Pose on func

        self.kimera_sync_pub= rospy.Publisher('/sync_semantic',)
        self.deeplab_pub = rospy.Publisher('/sim/mesh_map', PointCloud2, queue_size=10)

        self.finetune_pub= rospy.Publisher('/finetune_model', Finetune, queue_size=10)

        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map',Bool, queue_size=10 )
        
    def init_subscribers(self):
        """Initialize all subscribers"""

        # RGB+DEPTH(+Pose on func)
        scene_topic = rospy.get_param('~scene_topic', '/habitat/scene/sensors')
        self.rgb_sub = rospy.Subscriber(scene_topic,Sensors,self.sensors_env_callback)
        # From DeepLabV3
        rospy.Subscriber("/deeplab/segmented_image", Image, self.deeplab_labels_callback)
   

        
        # From Kimera
        rospy.Subscriber('/kimera/mesh_map', PointCloud2, self.mesh_map_callback)
        rospy.Subscriber('/kimera/labels', Image, self.kimera_labels_callback)
        
        
        
    def sensors_env_callback(self, msg):
        stamp = msg.header.stamp
        rospy.loginfo(f"Synchronized RGB + Depth @ {stamp}")

        # Save for later matching with segmentation result
        self.pending_frames[stamp.to_sec()] = (msg)

        # Send RGB to DeepLab for segmentation
        self.deeplab_pub.publish(msg.rgb)
        
    
        
    def mesh_map_callback(self, msg):
        self.current_mesh=msg
        
    def kimera_labels_callback(self, msg):
        """Callback for labels from Kimera"""
        # Process labels
        pass
        
    def deeplab_labels_callback(self, msg):
        stamp = msg.header.stamp.to_sec()

        if stamp in self.pending_frames:
            popmsg = self.pending_frames.pop(stamp)
            rospy.loginfo(f"Got segmented image @ {stamp}, matching RGB + Depth")
            res = SyncSemantic(
                header=std_msgs.msg.Header(stamp=rospy.Time.now()),
                rgb=popmsg.rgb,
                depth=popmsg.depth,
                labels=msg,
            )
            self.kimera_sync_pub.publish(res)
        else:
            rospy.logwarn(f"Segmented image with stamp {stamp} had no matching RGB/Depth pair")
        pass
        
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown() and self.running:
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass