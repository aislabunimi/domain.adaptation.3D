#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import tf_conversions
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from ros_deeplabv3.srv import FinetuneRequest
from kimera_interfacer.msg import SyncSemantic
from label_generator_ros.srv import InitLabelGenerator, GenerateLabel, GenerateLabelRequest
from LabelElaborator import LabelElaborator
from Modules import PILBridge
from metrics import SemanticsMeter
import tf2_ros
import imageio.v2 as imageio



class MockedControlNode:
    def __init__(self):
        rospy.init_node("Control_mock", anonymous=True)
        self.sub_factor=1
        # Publishers
        self.kimera_pub = rospy.Publisher('/sync_semantic', SyncSemantic, queue_size=10)
        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map', Bool, queue_size=1)
        self.depth_info_pub = rospy.Publisher("/depth_camera_info", CameraInfo, queue_size=1)
        self.miou_pub = rospy.Publisher('/miou', Float64, queue_size=1)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Params
        self.image_dir = rospy.get_param("~image_dir")
        self.depth_dir = rospy.get_param("~depth_dir")
        self.gt_label_dir = rospy.get_param("~gt_label_dir")
        self.dlab_label_dir = rospy.get_param("~dlab_label_dir")
        self.pose_dir = rospy.get_param("~pose_dir")
        self.int_dir = rospy.get_param("~intrinsic_dir")
        self.mesh_path = rospy.get_param("~mesh_path")
        self.serialized_path = rospy.get_param("~serialized_path")

        mapping = np.genfromtxt(rospy.get_param("~mapping_file"), delimiter=",")[1:, 1:4]
        self.class_colors = mapping
        self.label_elaborator = LabelElaborator(self.class_colors, confidence=0)
        self.meter_gt_dlab = SemanticsMeter(number_classes=40)
        self.meter_gt_pseudo = SemanticsMeter(number_classes=40)

        # Service clients
        rospy.wait_for_service('/label_generator/init')
        rospy.wait_for_service('/label_generator/generate')
        self.init_srv = rospy.ServiceProxy('/label_generator/init', InitLabelGenerator)
        self.generate_srv = rospy.ServiceProxy('/label_generator/generate', GenerateLabel)


    def publish_tf(self, pose, stamp):
        if len(pose) == 16:
            pose_matrix = np.array(pose).reshape(4, 4)
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "map"
            t.child_frame_id = "base_link_forward"
            t.transform.translation.x = pose_matrix[0, 3]
            t.transform.translation.y = pose_matrix[1, 3]
            t.transform.translation.z = pose_matrix[2, 3]
            q = tf_conversions.transformations.quaternion_from_matrix(pose_matrix)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)
        else:
            rospy.logwarn("Pose does not have 16 elements. Unable to publish transform.")

    def load_pose(self, path):
        with open(path, 'r') as f:
            return list(map(float, f.read().split()))

    def txt_to_camera_info(self, cam_p, img_p):
        data = np.loadtxt(cam_p)
        img = imageio.imread(img_p)
        msg = CameraInfo()
        msg.width = img.shape[1]
        msg.height = img.shape[0]
        msg.K = data[:3, :3].reshape(-1).tolist()
        msg.D = [0, 0, 0, 0, 0]
        msg.R = np.eye(3).reshape(-1).tolist()
        msg.P = data[:3, :4].reshape(-1).tolist()
        msg.distortion_model = "plumb_bob"
        return msg

    def rgb_to_class_index(self, rgb_image, tolerance=5):
        h, w = rgb_image.shape[:2]
        class_map = np.zeros((h, w), dtype=np.int32)
        for class_idx, color in enumerate(self.class_colors):
            mask = np.all(np.abs(rgb_image - color) <= tolerance, axis=-1)
            class_map[mask] = class_idx
        return class_map
    
    def load_all(self,rgb_image, depth_image, sem_image, map1, map2, sub_factor):
        """
        Loads and processes RGB, depth, and semantic images for mesh generation.

        Args:
            rgb_image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            sem_image (np.ndarray): Semantic image.
            map1 (np.ndarray): Remapping map for the RGB image.
            map2 (np.ndarray): Remapping map for the RGB image.
            sub_factor (int): Subsampling factor for depth image sparsification.

        Returns:
            tuple: Processed RGB, depth, and semantic images, aligned for mesh generation.
        """
        
        # Sparsify depth using sub_factor
        depth = depth_image.copy()
        mask = np.zeros_like(depth)
        mask[::sub_factor, ::sub_factor] = 1
        depth[mask != 1] = 0  # Set values not in the mask to 0

        # Remap RGB and Semantic images (depth doesn't need remapping)
        rgb = cv2.remap(rgb_image, map1, map2, interpolation=cv2.INTER_NEAREST)
        sem_img = cv2.remap(sem_image, map1, map2, interpolation=cv2.INTER_NEAREST)

        # Return the aligned images
        return rgb, depth.astype(np.int32), sem_img
    def run(self):
        rospy.loginfo("Preloading data...")

        img_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])

        rgb_path_pre = os.path.join(self.image_dir, img_files[0])
        rgb_image_pre = cv2.imread(rgb_path_pre)
        h,w = rgb_image_pre.shape[:2]

        k_image = [525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1]
        self.init_srv(h, w, k_image, self.mesh_path, self.serialized_path)

        # Preload only camera info (lightweight)
        camera_info_depth = self.txt_to_camera_info(os.path.join(self.int_dir, "intrinsic_depth.txt"), f"{self.image_dir}/0.jpg")
        camera_info_img = self.txt_to_camera_info(os.path.join(self.int_dir, "intrinsic_color.txt"), f"{self.depth_dir}/0.png")
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=np.array(camera_info_img.K).reshape(3, 3),
            distCoeffs=np.zeros(5),
            R=np.eye(3),
            newCameraMatrix=np.array(camera_info_depth.K).reshape(3, 3),
            size=(640, 480),  # target image size (typically depth resolution)
            m1type=cv2.CV_32FC1
        )
        rospy.loginfo("Sending RGB-D + Semantics to Kimera...")
        frame_idx=0
        for f in img_files:
            stamp = rospy.Time.now()

            # Update camera_info header
            camera_info_depth.header.stamp = stamp
            camera_info_depth.header.frame_id = "base_link_gt"
            self.depth_info_pub.publish(camera_info_depth)

            # Load pose (small file)
            pose_path = os.path.join(self.pose_dir, f.replace("frame", "pose").replace(".jpg", ".txt"))
            pose = self.load_pose(pose_path)
            self.publish_tf(pose, stamp)

            # Load and process RGB, Depth, and Semantic images using load_all
            rgb_path = os.path.join(self.image_dir, f)
            depth_path = os.path.join(self.depth_dir, f.replace("frame", "pose").replace(".jpg", ".png"))
            sem_path = os.path.join(self.dlab_label_dir, f)

            # Assuming rgb_image, depth_image, and sem_image are already read images
            rgb_image = cv2.imread(rgb_path)
          
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            sem_image = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
            _, colored_sem, _ = self.label_elaborator.process(sem_image)
            # Use load_all function to process the images
            rgb_processed, depth_processed, sem_processed = self.load_all(rgb_image, depth_image, colored_sem, map1, map2, self.sub_factor)

            gt = cv2.imread(os.path.join(self.gt_label_dir, f.replace("frame", "pose").replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)

            # Convert to ROS messages
            rgb_msg = PILBridge.PILBridge.numpy_to_rosimg(rgb_processed, encoding="rgb8")
            depth_msg = PILBridge.PILBridge.numpy_to_rosimg(depth_processed, encoding="16UC1")

            # Process DeepLab image
            
            sem_msg = PILBridge.PILBridge.numpy_to_rosimg(sem_processed, encoding="rgb8")

            # Publish semantic message
            for msg in (rgb_msg, depth_msg, sem_msg):
                msg.header.frame_id = "base_link_gt"
                msg.header.seq = frame_idx
                msg.header.stamp = stamp

            semantic = SyncSemantic()
            semantic.image = rgb_msg
            semantic.depth = depth_msg
            semantic.sem = sem_msg
            self.kimera_pub.publish(semantic)
            #print(f"[DEBUG] Frame {frame_idx}: RGB shape = {rgb.shape}, Depth shape = {depth.shape}, Sem shape = {colored_sem.shape}")
            # Evaluate prediction
            self.meter_gt_dlab.update(sem_image, gt)

            
            frame_idx += 1
            rospy.sleep(0.1)

        self.outmap_pub.publish(Bool(data=True))
        rospy.loginfo("Sending end of generation signal...")
        rospy.loginfo("Generating pseudo labels...")

        for i, fname in enumerate(img_files):
            pose_path = os.path.join(self.pose_dir, fname.replace("frame", "pose").replace(".jpg", ".txt"))
            pose = self.load_pose(pose_path)

            request = GenerateLabelRequest()
            request.pose = pose
            result = self.generate_srv(request)

            if not result.success:
                rospy.logerr(f"Label gen failed: {result.error_msg}")
                continue

            pseudo = PILBridge.PILBridge.rosimg_to_numpy(result.label)
            gt = cv2.imread(os.path.join(self.gt_label_dir, fname.replace("frame", "pose").replace(".jpg", ".png")))

            _, colored_gt, _ = self.label_elaborator.process(gt)
            _, colored_pseudo, _ = self.label_elaborator.process(pseudo)

            self.meter_gt_pseudo.update(self.rgb_to_class_index(colored_pseudo), self.rgb_to_class_index(colored_gt))

        miou_dlab, acc_dlab, class_acc_dlab = self.meter_gt_dlab.measure()
        miou_pseudo, acc_pseudo, class_acc_pseudo = self.meter_gt_pseudo.measure()

        rospy.loginfo(f"[DeepLab] mIoU: {miou_dlab:.3f}, Acc: {acc_dlab:.3f}, ClassAcc: {class_acc_dlab:.3f}")
        rospy.loginfo(f"[Pseudo]   mIoU: {miou_pseudo:.3f}, Acc: {acc_pseudo:.3f}, ClassAcc: {class_acc_pseudo:.3f}")
        self.miou_pub.publish(Float64(data=miou_pseudo))


if __name__ == "__main__":
    try:
        MockedControlNode().run()
    except rospy.ROSInterruptException:
        pass
