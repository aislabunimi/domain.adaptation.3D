#!/usr/bin/env python3
import rospy
import os
import cv2
import tf2_ros
import numpy as np
import tf_conversions
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Empty
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from ros_deeplabv3.srv import Finetune, FinetuneRequest
from kimera_interfacer.msg import SyncSemantic
from habitat_ros_bridge.msg import Sensors
from label_generator_ros.srv import InitLabelGenerator
from label_generator_ros.srv import GenerateLabel, GenerateLabelRequest, GenerateLabelResponse
from LabelElaborator import LabelElaborator
from Modules import PILBridge
from sensor_msgs.msg import CameraInfo


TEMP_DIR = "/home/michele/db"
os.makedirs(f"{TEMP_DIR}/images", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/labels", exist_ok=True)

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)
        self.label_elaborator = LabelElaborator(confidence=0)
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_store = {}  # timestamp: pose
        self.pending_frames = {}  # timestamp: Sensors msg

        self.image_id = 0
        self.current_mesh = "/home/michele/Desktop/Domain-Adaptation-Pipeline/Output_kimera_mesh/output_predict_mesh.ply"
        self.map_serialized_path = "/home/michele/Desktop/Domain-Adaptation-Pipeline/Output_kimera_mesh/output_serialized.data"
        self.width = None
        self.height = None
        self.k_image = None
        self.running = True

        self.init_publishers()
        self.init_subscribers()
        self.init_services()

    def init_publishers(self):
        self.kimera_sync_pub = rospy.Publisher('/sync_semantic', SyncSemantic, queue_size=10)
        self.deeplab_pub = rospy.Publisher('/deeplab/rgb', Image, queue_size=10)
        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map', Bool, queue_size=10)
        self.startexplore_pub = rospy.Publisher("/startexplore", Empty, queue_size=1)
        self.stopexplore_pub = rospy.Publisher("/stopexplore", Empty, queue_size=1)
        self.resetexplore_pub = rospy.Publisher("/resetexplore", Empty, queue_size=1)
        self.depth_info_pub=rospy.Publisher("/depth_camera_info", CameraInfo, queue_size=1)

    def init_subscribers(self):
        scene_topic = rospy.get_param('~scene_topic', '/habitat/scene/sensors')
        self.rgb_sub = rospy.Subscriber(scene_topic, Sensors, self.sensors_env_callback)
        rospy.Subscriber("/deeplab/segmented_image", Image, self.deeplab_labels_callback)
        rospy.Subscriber('/kimera/mesh_map', PointCloud2, self.mesh_map_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_color_callback)
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_depth_callback)

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
        stamp = msg.rgb.header.stamp
        rospy.loginfo(f"[Sensors] RGB + Depth received @ {stamp.to_sec()}")

        if self.width is None or self.height is None:
            self.width = msg.rgb.width
            self.height = msg.rgb.height

        # Save the message and pose for label association
        self.pending_frames[stamp.to_sec()] = msg
        self.pose_store[stamp.to_sec()] = msg.pose  # assume Sensors.msg has a .pose

        # Publish the current TF for Kimera to read
        self.publish_tf(msg.pose, stamp)

        # Send RGB to DeepLab
        self.deeplab_pub.publish(msg.rgb)

    def camera_info_color_callback(self, msg):
        if self.k_image is None:
            self.k_image = msg.K
    def camera_info_depth_callback(self, msg):
        self.depth_info_pub.publish(msg)

    def publish_tf(self, pose, stamp):
        # Assicurati che `pose` sia un array di float di dimensione 16 (4x4 appiattita)
        if len(pose) == 16:
            # Ricostruisci la matrice 4x4
            pose_matrix = np.array(pose).reshape(4, 4)

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "map"
            t.child_frame_id = "base_link_forward"

            # Estrai la traslazione (x, y, z) dalla matrice
            t.transform.translation.x = pose_matrix[0, 3]
            t.transform.translation.y = pose_matrix[1, 3]
            t.transform.translation.z = pose_matrix[2, 3]


            # Calcola il quaternione dalla matrice di rotazione
            q = tf_conversions.transformations.quaternion_from_matrix(pose_matrix)

            # Imposta i valori di rotazione (quaternione)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

        else:
            rospy.logwarn("Pose does not have 16 elements. Unable to publish transform.")


    def mesh_map_callback(self, msg):
        self.current_mesh = msg

    def deeplab_labels_callback(self, msg):
        stamp = msg.header.stamp.to_sec()

        if stamp in self.pending_frames:
            popmsg = self.pending_frames.pop(stamp)
            pose = self.pose_store.pop(stamp)
            rospy.loginfo(f"Matching segmented image @ {stamp}")

            # Convert ROS image to NumPy using PILBridge
            sem_cv = PILBridge.PILBridge.rosimg_to_numpy(msg)

            # Use LabelElaborator to process the NumPy image, not the ROS message
            _, colored_sem, _ = self.label_elaborator.process(sem_cv)

            # Convert colored semantic image back to ROS image
            sem_colored_msg = PILBridge.PILBridge.numpy_to_rosimg(colored_sem)
            sem_colored_msg.header = msg.header  # Preserve timestamp

            # Publish SyncSemantic
            semantic = SyncSemantic()
            semantic.image = popmsg.rgb
            semantic.depth = popmsg.depth
            semantic.sem = sem_colored_msg
            self.kimera_sync_pub.publish(semantic)

            # Save RGB image
            rgb_cv = PILBridge.PILBridge.rosimg_to_numpy(popmsg.rgb)
            rgb_path = f"{TEMP_DIR}/images/frame_{self.image_id:05d}.png"
            cv2.imwrite(rgb_path, rgb_cv)

            # Save pose
            self.save_pose(self.image_id, pose)

            self.image_id += 1
        else:
            rospy.logwarn(f"No RGB/depth pair found for label timestamp {stamp}")



    def save_pose(self, idx, pose):
        pose_path = os.path.join(TEMP_DIR, "poses", f"pose_{idx:05d}.txt")

        pose = np.array(pose).reshape(4, 4)
        
        # Assumiamo che `pose` sia un array numpy di dimensione (4, 4)
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            with open(pose_path, 'w') as f:
                # Scriviamo la matrice 4x4 riga per riga
                for row in pose:
                    # Scriviamo ogni riga della matrice come una stringa di float separati da spazi
                    f.write(" ".join(map(str, row)) + "\n")
        else:
            raise ValueError("pose must be a 4x4 numpy array")

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz = 1 iteration per second
        empty_msg = Empty()

        while not rospy.is_shutdown():
            rospy.loginfo("Starting exploration cycle...")

            self.startexplore_pub.publish(empty_msg)

            start_time = rospy.Time.now()

            # Let environment run for 40 seconds
            rospy.sleep(20.0)
            self.stopexplore_pub.publish(empty_msg)
            rospy.loginfo("Exploration stopping")

            rospy.sleep(10.0)

            self.resetexplore_pub.publish(empty_msg)
            rospy.loginfo("Resetting")


            # Stop Kimera & generate labels
            self.outmap_pub.publish(Bool(data=True))
            rospy.sleep(5.0)
            self.init_srv(
                int(self.height),
                int(self.width),
                [float(x) for x in self.k_image],
                self.current_mesh,
                self.map_serialized_path
            )           
            rospy.sleep(2.0)
            # Label generation for all saved poses
            for i in range(self.image_id):
                pose_path = os.path.join(TEMP_DIR, "poses", f"pose_{i:05d}.txt")
                if not os.path.exists(pose_path):
                    continue

                with open(pose_path, 'r') as f:
                    pose_data = list(map(float, f.read().split()))
                    if len(pose_data) != 16:
                        rospy.logwarn(f"Pose at {pose_path} does not have 16 elements.")
                        continue

                # Build request
                request = GenerateLabelRequest()
                request.pose = pose_data
                try:
                    result = self.generate_srv(request)
                    if result.success:
                        label_image = result.label
                        # You can now do something with the label image
                    else:
                        rospy.logerr(f"Label generation failed: {result.error_msg}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")

                # Save casted label
                label_cv = PILBridge.PILBridge.rosimg_to_numpy(result.label)
                label_path = f"{TEMP_DIR}/labels/label_{i:05d}.png"
                cv2.imwrite(label_path, label_cv)

            rospy.loginfo("Calling finetune service...")
            request = FinetuneRequest()
            request.dataset_path = "/home/michele/db"
            request.num_epochs = 10  # or however many epochs you want
            request.num_classes = 40
            try:
                response = self.finetune_srv(request)
                if response.success:
                    rospy.loginfo(f"Finetuning succeeded: {response.message}")
                else:
                    rospy.logerr(f"Finetuning failed: {response.message}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

            rospy.loginfo("Finetune completed. Starting new cycle.")
            rospy.sleep(3.0)

            rate.sleep()

if __name__ == '__main__':
    try:
        node = ControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

