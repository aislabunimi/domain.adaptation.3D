#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import tf_conversions
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from ros_deeplabv3.srv import FinetuneRequest
from kimera_interfacer.msg import SyncSemantic
from label_generator_ros.srv import InitLabelGenerator, GenerateLabel, GenerateLabelRequest
from LabelElaborator import LabelElaborator
from Modules import PILBridge
from sensor_msgs.msg import CameraInfo
from metrics import Metrics

class MockedControlNode:
    def __init__(self):
        rospy.init_node("mocked_control_node", anonymous=True)
        self.bridge = CvBridge()

        self.kimera_pub = rospy.Publisher('/sync_semantic', SyncSemantic, queue_size=10)
        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map', Bool, queue_size=1)
        self.miou_pub = rospy.Publisher('/miou', Float64, queue_size=1)

        self.image_dir = rospy.get_param("~image_dir")
        self.depth_dir = rospy.get_param("~depth_dir")
        self.gt_label_dir = rospy.get_param("~gt_label_dir")
        self.dlab_label_dir = rospy.get_param("~dlab_label_dir")
        self.pose_dir = rospy.get_param("~pose_dir")
        self.mesh_path = rospy.get_param("~mesh_path")
        self.serialized_path = rospy.get_param("~serialized_path")

        mapping = np.genfromtxt(rospy.get_param("~mapping_file"), delimiter=",")[1:, 1:4]
        self.class_colors = mapping
        self.label_elaborator = LabelElaborator(self.class_colors, confidence=0)
        self.meter_gt_dlab = Metrics(num_classes=40)
        self.meter_gt_pseudo = Metrics(num_classes=40)

        rospy.wait_for_service('/label_generator/init')
        rospy.wait_for_service('/label_generator/generate')
        self.init_srv = rospy.ServiceProxy('/label_generator/init', InitLabelGenerator)
        self.generate_srv = rospy.ServiceProxy('/label_generator/generate', GenerateLabel)

    def publish_tf(self, pose, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_link_forward"
        mat = np.array(pose).reshape(4, 4)
        t.transform.translation.x = mat[0, 3]
        t.transform.translation.y = mat[1, 3]
        t.transform.translation.z = mat[2, 3]
        q = tf_conversions.transformations.quaternion_from_matrix(mat)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_broadcaster.sendTransform(t)

    def load_pose(self, path):
        with open(path, 'r') as f:
            return list(map(float, f.read().split()))

    def rgb_to_class_index(self, rgb_image, tolerance=5):
        h, w, _ = rgb_image.shape
        class_map = np.zeros((h, w), dtype=np.int32)
        for class_idx, color in enumerate(self.class_colors):
            mask = np.all(np.abs(rgb_image - color) <= tolerance, axis=-1)
            class_map[mask] = class_idx
        return class_map

    def run(self):
        self.tf_broadcaster = rospy.Publisher("/tf", TransformStamped, queue_size=10)

        img_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
        for i, fname in enumerate(img_files):
            img_path = os.path.join(self.image_dir, fname)
            depth_path = os.path.join(self.depth_dir, fname)
            gt_label_path = os.path.join(self.gt_label_dir, fname)
            dlab_label_path = os.path.join(self.dlab_label_dir, fname)
            pose_path = os.path.join(self.pose_dir, fname.replace("frame", "pose").replace(".png", ".txt"))

            rgb = cv2.imread(img_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            pose = self.load_pose(pose_path)

            dlab = cv2.imread(dlab_label_path)
            gt = cv2.imread(gt_label_path)

            rgb_msg = PILBridge.PILBridge.numpy_to_rosimg(rgb)
            depth_msg = PILBridge.PILBridge.numpy_to_rosimg(depth)
            sem_msg = PILBridge.PILBridge.numpy_to_rosimg(dlab)

            semantic = SyncSemantic()
            semantic.image = rgb_msg
            semantic.depth = depth_msg
            semantic.sem = sem_msg
            self.kimera_pub.publish(semantic)

            pred = self.rgb_to_class_index(dlab)
            gt_idx = self.rgb_to_class_index(gt)
            self.meter_gt_dlab.update(pred, gt_idx)

        rospy.loginfo("Sending out_map trigger...")
        self.outmap_pub.publish(Bool(data=True))
        rospy.sleep(5.0)

        k_image = [525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1]  # Example intrinsics
        h, w = rgb.shape[:2]
        self.init_srv(h, w, k_image, self.mesh_path, self.serialized_path)

        for i, fname in enumerate(img_files):
            pose_path = os.path.join(self.pose_dir, fname.replace("frame", "pose").replace(".png", ".txt"))
            pose = self.load_pose(pose_path)
            request = GenerateLabelRequest()
            request.pose = pose
            result = self.generate_srv(request)

            if not result.success:
                rospy.logerr(f"Label gen failed: {result.error_msg}")
                continue

            pseudo = PILBridge.PILBridge.rosimg_to_numpy(result.label)
            gt = cv2.imread(os.path.join(self.gt_label_dir, fname))
            self.meter_gt_pseudo.update(self.rgb_to_class_index(pseudo), self.rgb_to_class_index(gt))

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
