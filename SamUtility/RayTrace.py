#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from TestScripts.Utilitity.PILBridge import PILBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from label_generator_ros.msg import LabelGenInit, LabelGenResponse  # Replace with actual package name
import matplotlib.pyplot as plt
from std_msgs.msg import Header, Float64MultiArray

class LabelGenTester:
    def __init__(self):
        rospy.init_node('label_gen_test_node')

        self.label_received = False
        self.label_image = None

        # Subscriber to get label image
        rospy.Subscriber('/label_generator/label', LabelGenResponse, self.label_callback)


        # Publisher to send request
        self.request_pub = rospy.Publisher('/label_generator/request', Float64MultiArray, queue_size=1)
        self.initreq_pub=rospy.Publisher('/label_generator/init',LabelGenInit,queue_size=1)
        rospy.sleep(1.0)  # Let ROS setup

        self.send_request()
        self.wait_for_response()
    
    # use later maybe? if i left this here good luck fixing this shit
    """def load_pose(self, path):
        pose_matrix = np.loadtxt(path).reshape((4, 4))
        pose = Pose()
        # Convert 4x4 matrix to Pose (rotation + translation)
        from tf.transformations import quaternion_from_matrix
        trans = pose_matrix[:3, 3]
        quat = quaternion_from_matrix(pose_matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose"""

    def send_request(self):
        req = LabelGenInit()
        ScannetDir = "/home/michele/Desktop/Colombo/Scannet/scans/scene0002_00"

        # Paths
        pose_path = f"{ScannetDir}/pose/0.txt"
        rgb_path = f"{ScannetDir}/color/0.jpg"
        k_image_path = f"{ScannetDir}/intrinsic/intrinsic_color.txt"
        req.mesh_path = "/home/michele/Desktop/Colombo/Scannet/output_kimera_semantics/scene0002_00_create_labels_predict_mesh.ply"
        req.map_serialized_path = "/home/michele/Desktop/Colombo/Scannet/output_kimera_semantics/scene0002_00_create_labels_serialized.data"

        try:
            

            rgb_cv = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = PILBridge.numpy_to_rosimg(rgb_cv, encoding='rgb8')
            req.width=rgb.width
            req.height=rgb.height
            # Load intrinsics
            data = np.loadtxt(k_image_path)
            k_image = np.array(
            [[data[0, 0], 0, data[0, 2]], [0, data[1, 1], data[1, 2]], [0, 0, 1]]
            )
            k = k_image.flatten()
            req.k_image = k.tolist()

            self.initreq_pub.publish(req)
            rospy.loginfo("Init request published.")

        except Exception as e:
            rospy.logerr(f"Error preparing or publishing request: {e}")

    def label_callback(self, msg):
        if( not msg.success):
            rospy.loginfo(msg.error_msg)
            return 
        
        rospy.loginfo("Label received.")
        self.label_image = PILBridge.rosimg_to_numpy(msg.label)
        self.label_received = True

    def wait_for_response(self):
        ScannetDir = "/home/michele/Desktop/Colombo/Scannet/scans/scene0002_00"
        pose_path = f"{ScannetDir}/pose/0.txt"
        rospy.sleep(60)
        pose_array = np.loadtxt(pose_path).flatten()

        msg = Float64MultiArray()
        msg.data = pose_array.tolist()  # Converti NumPy array in lista di float
        self.request_pub.publish(msg)
        timeout = rospy.Time.now() + rospy.Duration(120.0)
        rate = rospy.Rate(20)

        while not rospy.is_shutdown() and not self.label_received and rospy.Time.now() < timeout:
            rate.sleep()

        if self.label_image is not None:
            plt.imshow(self.label_image, cmap='gray', vmin=0, vmax=39)
            plt.title("Label Output")
            plt.axis('off')
            plt.show()
        else:
            rospy.logwarn("No label received within timeout.")

if __name__ == "__main__":
    try:
        LabelGenTester()
    except rospy.ROSInterruptException:
        pass
