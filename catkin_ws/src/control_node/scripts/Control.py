#!/usr/bin/env python3
import rospy
import os
import cv2
import tf2_ros
import tf_conversions
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from ros_deeplabv3 import Finetune
from kimera_interfacer.msg import SyncSemantic
from habitat_ros_bridge.msg import Sensors
from label_generator_ros.srv import InitLabelGenerator, GenerateLabel
from Modules import PILBridge

TEMP_DIR = "/home/michele/db"
os.makedirs(f"{TEMP_DIR}/images", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/labels", exist_ok=True)

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_store = {}  # timestamp: pose
        self.pending_frames = {}  # timestamp: Sensors msg

        self.image_id = 0
        self.current_mesh = None
        self.running = True

        self.init_publishers()
        self.init_subscribers()
        self.init_services()

    def init_publishers(self):
        self.kimera_sync_pub = rospy.Publisher('/sync_semantic', SyncSemantic, queue_size=10)
        self.deeplab_pub = rospy.Publisher('/sim/mesh_map', PointCloud2, queue_size=10)
        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map', Bool, queue_size=10)

    def init_subscribers(self):
        scene_topic = rospy.get_param('~scene_topic', '/habitat/scene/sensors')
        self.rgb_sub = rospy.Subscriber(scene_topic, Sensors, self.sensors_env_callback)
        rospy.Subscriber("/deeplab/segmented_image", Image, self.deeplab_labels_callback)
        rospy.Subscriber('/kimera/mesh_map', PointCloud2, self.mesh_map_callback)

    def init_services(self):
        rospy.loginfo("Waiting for services...")
        rospy.wait_for_service('/deeplab/finetune_model')
        rospy.wait_for_service('/label_generator/init')
        rospy.wait_for_service('/label_generator/generate')

        self.finetune_srv = rospy.ServiceProxy('/deeplab/finetune_model', Finetune)
        self.init_srv = rospy.ServiceProxy('/label_generator/init', InitLabelGenerator)
        self.generate_srv = rospy.ServiceProxy('/label_generator/generate', GenerateLabel)
        rospy.loginfo("All services initialized.")

    def sensors_env_callback(self, msg):
        stamp = msg.header.stamp
        rospy.loginfo(f"[Sensors] RGB + Depth received @ {stamp.to_sec()}")

        # Save the message and pose for label association
        self.pending_frames[stamp.to_sec()] = msg
        self.pose_store[stamp.to_sec()] = msg.pose  # assume Sensors.msg has a .pose

        # Publish the current TF for Kimera to read
        self.publish_tf(msg.pose, stamp)

        # Send RGB to DeepLab
        self.deeplab_pub.publish(msg.rgb)

    def publish_tf(self, pose, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def mesh_map_callback(self, msg):
        self.current_mesh = msg

    def deeplab_labels_callback(self, msg):
        stamp = msg.header.stamp.to_sec()

        if stamp in self.pending_frames:
            popmsg = self.pending_frames.pop(stamp)
            pose = self.pose_store.pop(stamp)
            rospy.loginfo(f"Matching segmented image @ {stamp}")

            semantic = SyncSemantic()
            semantic.header.stamp = rospy.Time.now()
            semantic.rgb = popmsg.rgb
            semantic.depth = popmsg.depth
            semantic.labels = msg
            self.kimera_sync_pub.publish(semantic)

            # Save RGB image to disk
            rgb_cv = self.bridge.imgmsg_to_cv2(popmsg.rgb, "bgr8")
            rgb_path = f"{TEMP_DIR}/images/frame_{self.image_id:05d}.png"
            cv2.imwrite(rgb_path, rgb_cv)

            # Save pose for label casting
            self.save_pose(self.image_id, pose)

            self.image_id += 1
        else:
            rospy.logwarn(f"No RGB/depth pair found for label timestamp {stamp}")

    def save_pose(self, idx, pose):
        pose_path = os.path.join(TEMP_DIR, "images", f"pose_{idx:05d}.txt")
        with open(pose_path, 'w') as f:
            f.write(f"{pose.position.x} {pose.position.y} {pose.position.z} "
                    f"{pose.orientation.x} {pose.orientation.y} "
                    f"{pose.orientation.z} {pose.orientation.w}")

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz = 1 iteration per second
        self.init_srv()  # Initialize label generator
        rospy.sleep(2.0)

        while not rospy.is_shutdown():
            rospy.loginfo("Starting exploration cycle...")
            start_time = rospy.Time.now()

            # Let environment run for 40 seconds
            rospy.sleep(40.0)

            # Stop Kimera & generate labels
            self.outmap_pub.publish(Bool(data=True))
            rospy.sleep(5.0)

            if self.current_mesh is None:
                rospy.logwarn("No mesh available after exploration.")
                continue

            # Label generation for all saved poses
            for i in range(self.image_id):
                pose_path = os.path.join(TEMP_DIR, "images", f"pose_{i:05d}.txt")
                if not os.path.exists(pose_path):
                    continue

                with open(pose_path, 'r') as f:
                    data = list(map(float, f.read().split()))
                    position = data[0:3]
                    orientation = data[3:7]

                # Create and call the label generation service
                request = GenerateLabel._request_class()
                request.pose.position.x, request.pose.position.y, request.pose.position.z = position
                request.pose.orientation.x, request.pose.orientation.y, request.pose.orientation.z, request.pose.orientation.w = orientation

                result = self.generate_srv(request)

                # Save casted label
                label_cv = self.bridge.imgmsg_to_cv2(result.label, "mono8")
                label_path = f"{TEMP_DIR}/labels/label_{i:05d}.png"
                cv2.imwrite(label_path, label_cv)

            rospy.loginfo("Calling finetune service...")
            self.finetune_srv()
            rospy.loginfo("Finetune completed. Starting new cycle.")
            rospy.sleep(3.0)

            rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
