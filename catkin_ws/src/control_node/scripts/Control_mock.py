#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import tf_conversions
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float64, Float32
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from ros_deeplabv3.srv import FinetuneRequest
from kimera_interfacer.msg import SyncSemantic
from label_generator_ros.srv import InitLabelGenerator, GenerateLabel, GenerateLabelRequest
from LabelElaborator import LabelElaborator
from Modules import PILBridge
import time
from metrics import SemanticsMeter
from collections import Counter
import tf2_ros
from tqdm import tqdm
import imageio.v2 as imageio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from AutomaticSamRefiner import FastSamRefinerAuto
from FastSamRefiner import SAM2RefinerFast
from SamMetrics import SamMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import sys

timer = np.float32(0.0)

class MockedControlNode:

    def __init__(self,auto_yes=False):
        self.auto_yes = auto_yes
        rospy.init_node("Control_mock", anonymous=True)
        self.sub_factor=1
        self.original_image_size=(640,480)
        # Publishers
        self.kimera_pub = rospy.Publisher('/sync_semantic', SyncSemantic, queue_size=10)
        self.outmap_pub = rospy.Publisher('/kimera/end_generation_map', Bool, queue_size=1)
        self.depth_info_pub = rospy.Publisher("/depth_camera_info", CameraInfo, queue_size=1)
        self.miou_pub = rospy.Publisher('/miou', Float64, queue_size=1)
        self.ray_cast_pub = rospy.Publisher('/rayCasted', Image, queue_size=1)
        self.label_nyu40_pub = rospy.Publisher('/label_nyu40', Image, queue_size=1)
        

        # Susbs
        rospy.Subscriber("/kimera/integration_duration", Float32, self.timer_callback)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Params
        self.image_size=(rospy.get_param("~img_size_w"),rospy.get_param("~img_size_h"))
        self.image_dir = rospy.get_param("~image_dir")
        self.depth_dir = rospy.get_param("~depth_dir")
        self.gt_label_dir = rospy.get_param("~gt_label_dir")
        self.dlab_label_dir = rospy.get_param("~dlab_label_dir")
        self.pose_dir = rospy.get_param("~pose_dir")
        self.int_dir = rospy.get_param("~intrinsic_dir")
        self.mesh_path = rospy.get_param("~mesh_path")
        self.serialized_path = rospy.get_param("~serialized_path")
        self.scene_number = rospy.get_param("~scene_number")
        self.pseudo_dir=rospy.get_param("~pseudo_dir")
        self.sam_dir=rospy.get_param("~sam_dir")
        self.automatic = rospy.get_param("~automatic", True)  # Default to False if not set

        mapping = np.genfromtxt(rospy.get_param("~mapping_file"), delimiter=",")[1:, 1:4]
        self.class_colors = mapping
        self.label_elaborator = LabelElaborator(self.class_colors, confidence=0)
        self.meter_gt_dlab = SemanticsMeter(number_classes=40)
        self.meter_gt_pseudo = SemanticsMeter(number_classes=40)
        self.meter_gt_sam=SemanticsMeter(number_classes=40)
        # Service clients
        rospy.wait_for_service('/label_generator/init')
        rospy.wait_for_service('/label_generator/generate')
        self.init_srv = rospy.ServiceProxy('/label_generator/init', InitLabelGenerator)
        self.generate_srv = rospy.ServiceProxy('/label_generator/generate', GenerateLabel)

    # region Utils

    def save_confusion_matrix_txt(self, conf_matrix, output_path="confusion_matrix.txt"):
        with open(output_path, 'w') as f:
            f.write("Confusion Matrix (Refined Class = Rows, Original Class = Columns):\n")
            for row in conf_matrix:
                row_str = "\t".join(f"{val:5d}" for val in row)
                f.write(row_str + "\n")

    def save_confusion_matrix_image(self, conf_matrix, output_path="confusion_matrix.pdf"):
        # Normalize to percentages
        total = conf_matrix.sum()
        if total > 0:
            conf_percent = (conf_matrix / total) * 100.0
        else:
            conf_percent = conf_matrix.copy()

        # Identify rows and columns that are not entirely zero
        valid_rows = np.any(conf_percent != 0, axis=1)
        valid_cols = np.any(conf_percent != 0, axis=0)

        # Filter matrix and corresponding labels
        filtered_matrix = conf_percent[valid_rows][:, valid_cols]
        row_labels = np.arange(conf_matrix.shape[0])[valid_rows]
        col_labels = np.arange(conf_matrix.shape[1])[valid_cols]

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            filtered_matrix,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar_kws={'label': 'Change %'},
            square=True,
            linewidths=.5,
            xticklabels=col_labels,
            yticklabels=row_labels
        )

        plt.title("SAM Refinement Confusion Matrix (Zero-only Rows/Cols Removed)")
        plt.xlabel("Original Class")
        plt.ylabel("Refined Class")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def timer_callback(self, msg):
        global timer
        timer = np.float32(msg.data)

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
    
    def load_pose(self, path):
        with open(path, 'r') as f:
            return list(map(float, f.read().split()))

    def txt_to_camera_info(self, cam_p, img_p, target_width, target_height):
        data = np.loadtxt(cam_p)
        img = imageio.imread(img_p)
        original_height, original_width = img.shape[:2]

        # Compute scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Scale intrinsics
        K = data[:3, :3].copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy

         # Scale intrinsics
        P = data[:3, :4].copy()
        P[0, 0] *= scale_x  # fx
        P[1, 1] *= scale_y  # fy
        P[0, 2] *= scale_x  # cx
        P[1, 2] *= scale_y  # cy

        # Create and fill CameraInfo message
        msg = CameraInfo()
        msg.width = target_width
        msg.height = target_height
        msg.K = K.reshape(-1).tolist()
        msg.D = [0, 0, 0, 0, 0]
        msg.R = np.eye(3).reshape(-1).tolist()
        msg.P = P.reshape(-1).tolist()
        msg.distortion_model = "plumb_bob"

        return msg
    
    def load_data(self, f, width, height, map1_rgb, map2_rgb, map1_depth, map2_depth):
        # Build full paths
        rgb_path = os.path.join(self.image_dir, f)
        depth_path = os.path.join(self.depth_dir, f.replace("frame", "pose").replace(".jpg", ".png"))
        sem_path = os.path.join(self.dlab_label_dir, f.replace(".jpg", ".png"))
        gt_path = os.path.join(self.gt_label_dir, f.replace("frame", "pose").replace(".jpg", ".png"))

        # Load images
        rgb_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        sem_image = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        # Remap images directly to target (width, height)
        rgb_image = cv2.remap(rgb_image, map1_rgb, map2_rgb, interpolation=cv2.INTER_LINEAR)
        #sem_image = cv2.remap(sem_image, map1_rgb, map2_rgb, interpolation=cv2.INTER_NEAREST) ????
        depth_image = cv2.remap(depth_image, map1_depth, map2_depth, interpolation=cv2.INTER_NEAREST)
        gt_image = cv2.resize(gt_image, (width, height), interpolation=cv2.INTER_NEAREST)  # Resize if needed

        # Generate colored semantic
        _, colored_sem, _ = self.label_elaborator.process(sem_image)

        return rgb_image, depth_image, sem_image, colored_sem, gt_image
        
    def getmaps(self,target_width,target_height):

        """
        Computes undistort-rectify maps and scaled intrinsics for both RGB and Depth cameras.
        Returns:
            map1_rgb, map2_rgb: Remapping maps for RGB/semantic images
            map1_depth, map2_depth: Remapping maps for depth images
            K_rgb_scaled: Scaled intrinsic matrix for RGB
            K_depth_scaled: Scaled intrinsic matrix for Depth
        """
        # Load original intrinsics
        K_rgb = np.loadtxt(os.path.join(self.int_dir, "intrinsic_color.txt"))[:3, :3]
        K_depth = np.loadtxt(os.path.join(self.int_dir, "intrinsic_depth.txt"))[:3, :3]

        # Define original resolutions
        orig_rgb_size = (1296, 968)
        orig_depth_size = (640, 480)


        # --- Scale intrinsics ---
        def scale_K(K, orig_size, target_size):
            scale_x = target_size[0] / orig_size[0]
            scale_y = target_size[1] / orig_size[1]
            K_scaled = K.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
            return K_scaled

        K_rgb_scaled = scale_K(K_rgb, orig_rgb_size, self.image_size)
        K_depth_scaled = scale_K(K_depth, orig_depth_size, self.image_size)

        # --- Compute rectification maps ---
        map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(
            cameraMatrix=K_rgb,
            distCoeffs=np.zeros(5),
            R=np.eye(3),
            newCameraMatrix=K_rgb_scaled,
            size=self.image_size,
            m1type=cv2.CV_32FC1
        )

        map1_depth, map2_depth = cv2.initUndistortRectifyMap(
            cameraMatrix=K_depth,
            distCoeffs=np.zeros(5),
            R=np.eye(3),
            newCameraMatrix=K_depth_scaled,
            size=self.image_size,
            m1type=cv2.CV_32FC1
        )

        return map1_rgb, map2_rgb, map1_depth, map2_depth

    # endregion
    
    # region metrics
    def calculate_metrics(self,pred_dir, gt_dir, meter, resize_to=(320, 240), file_ext=".png",perc=0.8):
        """
        Calculates mIoU, pixel accuracy, and per-class accuracy between prediction and ground truth labels.

        Args:
            pred_dir (str): Directory with predicted label images.
            gt_dir (str): Directory with ground truth label images.
            meter (object): Metric meter with .reset(), .update(pred, gt), and .measure() -> (miou, acc, class_acc).
            resize_to (tuple): Target image size (width, height).
            file_ext (str): Extension of label images (e.g., '.png').

        Returns:
            tuple: (miou: float, accuracy: float, per_class_accuracy: np.ndarray)
        """
        #return 0, 0, 0
        pred_files = sorted(
            [f for f in os.listdir(pred_dir) if f.endswith(file_ext)],
            key=lambda x: int(x.split('.')[0])
        )
        missing_classes_counter = Counter()
        missing_class_counts = []
        meter.clear()  # Clear previous state
        num_files = int(len(pred_files) * perc)
        for f in tqdm(pred_files[:num_files], desc="Evaluating metrics"):
            pred_path = os.path.join(pred_dir, f)
            gt_path = os.path.join(gt_dir, f)

            pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

            if pred_img is None:
                print(f"[WARNING] Could not read prediction image: {pred_path}")
                continue
            if gt_img is None:
                print(f"[WARNING] Could not read ground truth image: {gt_path}")
                continue

            

            # Resize prediction if needed
            if pred_img.shape[:2] != resize_to[::-1]:
                #print(f"[INFO] Resizing prediction {f} from {pred_img.shape[::-1]} to {resize_to}")
                rospy.loginfo_once(f"[INFO] Resizing prediction {f} from {pred_img.shape[::-1]} to {resize_to}")
                pred_img = cv2.resize(pred_img, resize_to, interpolation=cv2.INTER_NEAREST)

            # Resize ground truth if needed
            if gt_img.shape[:2] != resize_to[::-1]:
                #print(f"[INFO] Resizing ground truth {f} from {gt_img.shape[::-1]} to {resize_to}")
                rospy.loginfo_once(f"[INFO] Resizing ground truth {f} from {gt_img.shape[::-1]} to {resize_to}")
                gt_img = cv2.resize(gt_img, resize_to, interpolation=cv2.INTER_NEAREST)

            # Convert to int and shift class IDs if needed
            if len(pred_img.shape) == 3 and pred_img.shape[2] == 3:
                # Image is color (likely RGB), so convert to class index
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                pred_img = self.rgb_to_class_index(pred_img)
            else:
                # Image is grayscale or single channel, skip or handle differently if needed
                rospy.logwarn_once("Prediction image is grayscale, skipping rgb_to_class_index conversion.")
            pred_img = pred_img.astype(np.int16) - 1
            gt_img = gt_img.astype(np.int16) - 1

            # region Debug
            gt_classes = set(np.unique(gt_img)) - {-1}
            pred_classes = set(np.unique(pred_img)) - {-1}
            missing_classes = gt_classes - pred_classes
            missing_class_counts.append(len(missing_classes))
            missing_classes_counter.update(missing_classes)

            # endregion

            if np.all(gt_img == -1) or np.all(pred_img == -1):
                rospy.logwarn(f"Skipping {f} because prediction or GT is fully void.")
                continue
            
            meter.update(pred_img, gt_img)

        # region Debug
        print("\n[STATISTICS] Missing Class Analysis:")
        print(f"- Median number of missing classes per image: {np.median(missing_class_counts):.1f}")
        total_images = len(missing_class_counts)
        for cls, count in sorted(missing_classes_counter.items()):
            pct = 100.0 * count / total_images
            print(f"  - Class {cls}: missed in {count} images ({pct:.1f}%)")
        # endregion
        
        miou, acc, class_acc = meter.measure()
        return miou, acc, class_acc    
        
    def calculate_sam_metrics(self,pred_dir, pseudo_dir,gt_dir, log_path="sam_metrics_log.json", resize_to=(320, 240), file_ext=".png"):
        """
        Calculates refinement statistics using SamMetrics by comparing original and refined label maps.

        Args:
            pred_dir (str): Directory with refined label images (after SAM).
            gt_dir (str): Directory with original label images.
            log_path (str): Path to log file for per-frame metrics.
            resize_to (tuple): Target image size (width, height).
            file_ext (str): Image file extension (e.g., '.png').
            perc (float): Fraction of files to process.

        Returns:
            tuple: (median_pixel_change: float, global_confusion_matrix: np.ndarray)
        """
        
        pred_files = sorted(
            [f for f in os.listdir(pred_dir) if f.endswith(file_ext)],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        sam_meter = SamMetrics(log_path, True)
        num_files = int(len(pred_files))
        
        for f in tqdm(pred_files[:num_files], desc="Evaluating SAM refinements"):
            pred_path = os.path.join(pred_dir, f)
            pseudo_path = os.path.join(pseudo_dir, f)
            gt_path=os.path.join(gt_dir, f)

            pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            pseudo_img = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)
            gt_img=cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

            pseudo_img = cv2.cvtColor(pseudo_img, cv2.COLOR_BGR2RGB)
            pseudo_img = self.rgb_to_class_index(pseudo_img)

            if pred_img is None:
                print(f"[WARNING] Could not read prediction image: {pred_path}")
                continue
            if gt_img is None:
                print(f"[WARNING] Could not read ground truth image: {pseudo_path}")
                continue
            if pseudo_img is None:
                print(f"[WARNING] Could not read pseudo image: {pseudo_path}")
                continue

            if pred_img.shape[:2] != resize_to[::-1]:
                rospy.loginfo_once(f"[INFO] Resizing prediction {f} from {pred_img.shape[::-1]} to {resize_to}")
                pred_img = cv2.resize(pred_img, resize_to, interpolation=cv2.INTER_NEAREST)

            if gt_img.shape[:2] != resize_to[::-1]:
                rospy.loginfo_once(f"[INFO] Resizing GT {f} from {gt_img.shape[::-1]} to {resize_to}")
                gt_img = cv2.resize(gt_img, resize_to, interpolation=cv2.INTER_NEAREST)
            
            if pseudo_img.shape[:2] != resize_to[::-1]:
                rospy.loginfo_once(f"[INFO] Resizing GT {f} from {pseudo_img.shape[::-1]} to {resize_to}")
                pseudo_img = cv2.resize(pseudo_img, resize_to, interpolation=cv2.INTER_NEAREST)

            pred_img = pred_img.astype(np.int16) - 1
            gt_img = gt_img.astype(np.int16) - 1
            pseudo_img=pseudo_img.astype(np.int16) -1

            if np.all(gt_img == -1) or np.all(pred_img == -1):
                rospy.logwarn(f"Skipping {f} because prediction or GT is fully void.")
                continue

            sam_meter.update(frame_id=f, info=self.scene_number, original=pseudo_img, gt=gt_img, refined=pred_img)

        median_change, global_conf_matrix = sam_meter.measure()
        sam_meter.save_log()
        self.save_confusion_matrix_image(global_conf_matrix, output_path=os.path.join(pred_dir, "sam_confusion_matrix.pdf"))
        self.save_confusion_matrix_txt(global_conf_matrix, output_path=os.path.join(pred_dir, "sam_confusion_matrix.txt"))
        return median_change, global_conf_matrix
    # endregion

    # region generators
    def kimera_mesh_generator(self):

        rospy.loginfo("Preloading data for mesh generation...")

        img_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")],
                   key=lambda x: int(x.split('.')[0]))
       
        camera_info_depth = self.txt_to_camera_info(os.path.join(self.int_dir, "intrinsic_depth.txt"), f"{self.depth_dir}/0.png", *self.image_size)
        map1_rgb, map2_rgb, map1_depth, map2_depth=self.getmaps(*self.image_size)

        rospy.loginfo("Sending RGB-D + Semantics to Kimera...")
        for frame_idx, f in enumerate(tqdm(img_files, desc="Sending frames to Kimera")):
            
            stamp = rospy.Time.now()
            
            camera_info_depth.header.stamp = stamp
            camera_info_depth.header.frame_id = "base_link_gt"
            self.depth_info_pub.publish(camera_info_depth)

            pose_path = os.path.join(self.pose_dir, f.replace("frame", "pose").replace(".jpg", ".txt"))
            pose = self.load_pose(pose_path)
            if np.any(np.isinf(pose)):
                rospy.logwarn("Pose contains infinite values, skipping this pose: %s", pose_path)
                continue
            
            self.publish_tf(pose, stamp)

            rgb_processed,depth_processed,sem_processed,sem_color_processed, gt_processed=self.load_data(f,*self.image_size, map1_rgb, map2_rgb, map1_depth, map2_depth)
            rgb_msg = PILBridge.PILBridge.numpy_to_rosimg(rgb_processed, encoding="rgb8")
            depth_msg = PILBridge.PILBridge.numpy_to_rosimg(depth_processed, encoding="16UC1")
            sem_msg = PILBridge.PILBridge.numpy_to_rosimg(sem_color_processed, encoding="rgb8")

            for msg in (rgb_msg, depth_msg, sem_msg):
                msg.header.frame_id = "base_link_gt"
                msg.header.seq = frame_idx
                msg.header.stamp = stamp

            semantic = SyncSemantic()
            semantic.image = rgb_msg
            semantic.depth = depth_msg
            semantic.sem = sem_msg
            self.kimera_pub.publish(semantic)
            sem_processed = sem_processed.astype(np.int16) - 1
            gt_processed = gt_processed.astype(np.int16) - 1

            self.meter_gt_dlab.update(sem_processed, gt_processed)
            rospy.sleep(1)

        rospy.sleep(10)

        rospy.loginfo("Sending end of generation signal...")
        self.outmap_pub.publish(Bool(data=True))
        rospy.sleep(120)

    def pseudo_label_generator(self):

        rospy.loginfo("Generating pseudo labels with ray tracing...")

        w,h = self.image_size

        img_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")],
                   key=lambda x: int(x.split('.')[0]))
        
        camera_info = self.txt_to_camera_info(os.path.join(self.int_dir, "intrinsic_color.txt"), f"{self.image_dir}/0.jpg",w,h)

        self.init_srv(h, w, np.array(camera_info.K), self.mesh_path, self.serialized_path)

        
            # Create output directory if it doesn't exist
        if not os.path.exists(self.pseudo_dir):
            os.makedirs(self.pseudo_dir)

        for i, fname in enumerate(tqdm(img_files, desc="Generating pseudo labels")):

            pose_path = os.path.join(self.pose_dir, fname.replace("frame", "pose").replace(".jpg", ".txt"))
            pose = self.load_pose(pose_path)
            if np.any(np.isinf(pose)):
                rospy.logwarn("Pose contains infinite values, skipping this pose: %s", pose_path)
                continue
            
            request = GenerateLabelRequest()
            request.pose = pose
            result = self.generate_srv(request)
            if not result.success:
                rospy.logerr(f"Label gen failed: {result.error_msg}")
                continue
            
            pseudo = PILBridge.PILBridge.rosimg_to_numpy(result.label)
            gt_path = os.path.join(self.gt_label_dir, fname.replace("frame", "pose").replace(".jpg", ".png"))
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            
            _, colored_gt, _ = self.label_elaborator.process(gt)
            _, colored_pseudo, _ = self.label_elaborator.process(pseudo)




            output_filename = fname.replace(".jpg", ".png")
            output_path = os.path.join(self.pseudo_dir, output_filename)

            colored_pseudo_bgr = cv2.cvtColor(colored_pseudo, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, colored_pseudo_bgr)
            colored_gt_msg = PILBridge.PILBridge.numpy_to_rosimg(colored_gt)
            colored_pseudo_msg = PILBridge.PILBridge.numpy_to_rosimg(colored_pseudo)
            self.ray_cast_pub.publish(colored_pseudo_msg)
            self.label_nyu40_pub.publish(colored_gt_msg)

    def refine_with_sam(self, pseudo_label_files, resize_to=(320, 240)):
        """
        Refines all pseudo-labels using SAM2RefinerMixed and saves results to sam_refined_dir.
        Assumes each pseudo label has a matching RGB image with the same filename.

        Args:
            pseudo_label_files (List[str]): List of paths to pseudo-label PNGs.
            resize_to (Tuple[int, int]): Resize (width, height) for both image and mask before refinement. Default: (320, 240)
        """
        rospy.loginfo("Initializing SAM2RefinerMixed...")
        
        if self.automatic:
            granularity = max(1, resize_to[0] // 10)
            refiner = FastSamRefinerAuto(visualize=False, granularity=granularity)#CHANGE TO CHECK RESULTS
        else:
            refiner = SAM2RefinerFast(
                visualize=False,
                batch_size=16,
                skip_labels=None,
                fill_strategy="ereditary",
                skip_max_labels=[1],
                min_area_ratio=0.001
            )
        size_str = f"_{resize_to[0]}x{resize_to[1]}"
        samDir = self.sam_dir + ("_auto" if self.automatic else "_prompt") + size_str
        os.makedirs(samDir, exist_ok=True)
        target_w, target_h = resize_to

        for label_path in tqdm(pseudo_label_files, desc="Refining masks with SAM"):
            filename = os.path.basename(label_path)
            image_path = os.path.join(self.image_dir, filename.replace(".png", ".jpg"))

            if not os.path.exists(image_path):
                rospy.logwarn(f"RGB image not found for label '{filename}', skipping...")
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(label_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            if image is None or mask is None:
                rospy.logwarn(f"Failed to load image or mask for '{filename}', skipping...")
                continue

            # Resize both image and mask to target size
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if mask.shape[:2] != (target_h, target_w):
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            mask = self.rgb_to_class_index(mask)

            refined = refiner.refine(image, mask)

            save_path = os.path.join(samDir, filename)
            cv2.imwrite(save_path, refined)

        rospy.loginfo("All SAM refinements completed.")

    # endregion

    def run(self):
        # Step 1: Handle mesh
        if os.path.exists(self.mesh_path):
            rospy.logwarn(f"Mesh file '{self.mesh_path}' already exists.")
            try:
                if self.auto_yes:
                    answer = "y"
                else:
                    answer="n"
                    #answer = input("Mesh already exists. Regenerate? [y/N]: ").strip().lower()
            except EOFError:
                rospy.logerr("Cannot ask for user input. Running in non-interactive mode. Skipping mesh regeneration.")
                answer = "n"

            if answer == "y":
                self.kimera_mesh_generator()
            else:
                rospy.loginfo("Skipping Kimera interfacing...")
        else:
            self.kimera_mesh_generator()

        # Step 2: Handle pseudo-labels directory
        pseudo_dir = self.pseudo_dir
        if os.path.isdir(pseudo_dir) and os.listdir(pseudo_dir):  # Directory not empty
            try:
                if self.auto_yes:
                    answer = "y"
                else:
                    answer="n"
                    #answer = input("PseudoLabels directory is not empty. Regenerate? [y/N]: ").strip().lower()
            except EOFError:
                rospy.logerr("Cannot ask for user input. Running in non-interactive mode. Skipping pseudo label generation.")
                answer = "n"

            if answer == "y":
                rospy.loginfo("Clearing pseudo-labels directory...")
                for file in os.listdir(pseudo_dir):
                    file_path = os.path.join(pseudo_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                self.pseudo_label_generator()
            else:
                rospy.loginfo("Skipping pseudo-label generation.")
        else:
            self.pseudo_label_generator()

        # Step 2.5: Refine pseudo-labels with SAM
        resize_to=(1296,968) #1296x968  320x240
        size_str = f"_{resize_to[0]}x{resize_to[1]}"
        sam_refined_dir = self.sam_dir + ("_auto" if self.automatic else "_prompt") + size_str # You should define this in __init__ or elsewhere
        
        fps=1
        if os.path.isdir(sam_refined_dir) and os.listdir(sam_refined_dir):  # Directory exists and is not empty
            try:
                if self.auto_yes:
                    answer = "y"
                else:
                    answer="y"
                    #answer = input("SAM refined directory is not empty. Regenerate? [y/N]: ").strip().lower()
            except EOFError:
                rospy.logerr("Cannot ask for user input. Running in non-interactive mode. Skipping SAM refinement.")
                answer = "n"

            if answer == "y":
                rospy.loginfo("Clearing SAM refined directory...")
                for file in os.listdir(sam_refined_dir):
                    file_path = os.path.join(sam_refined_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                rospy.loginfo("Refining pseudo-labels with SAM...")
                pseudo_label_files = [
                os.path.join(pseudo_dir, f)
                for f in sorted(
                    [f for f in os.listdir(pseudo_dir) if f.endswith(".png")],
                    key=lambda x: int(os.path.splitext(x)[0])
                )[::fps]
                ]
                self.refine_with_sam(pseudo_label_files,resize_to)
            else:
                rospy.loginfo("Skipping SAM refinement.")
        else:
            rospy.loginfo("SAM refined directory is empty or does not exist. Refining pseudo-labels with SAM...")
            pseudo_label_files = [
            os.path.join(pseudo_dir, f)
            for f in sorted(
                [f for f in os.listdir(pseudo_dir) if f.endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0])
            )[::fps]
            ]
            self.refine_with_sam(pseudo_label_files,resize_to)
        
        # Step 3: Evaluate metrics after pseudo label generation and SAM refinement
        
        
        miou_dlab, acc_dlab, class_acc_dlab = self.calculate_metrics(
            pred_dir=self.dlab_label_dir,
            gt_dir=self.gt_label_dir,
            meter=self.meter_gt_dlab
        )

        miou_pseudo, acc_pseudo, class_acc_pseudo = self.calculate_metrics(
            pred_dir=pseudo_dir,
            gt_dir=self.gt_label_dir,
            meter=self.meter_gt_pseudo
        )

        miou_sam, acc_sam, class_acc_sam = self.calculate_metrics(
            pred_dir=sam_refined_dir,
            gt_dir=self.gt_label_dir,
            meter=self.meter_gt_sam , perc=0.8 # Make sure this exists in your class!
        )
        
        # Step 3.5: Evaluate change metrics with SamMetrics
        sam_avg, sam_conf = self.calculate_sam_metrics(
            pred_dir=sam_refined_dir,
            pseudo_dir=self.pseudo_dir,
            gt_dir=self.gt_label_dir,
            resize_to=(320, 240),
            log_path=sam_refined_dir+"/log.txt"
        )

        # Step 4: Log results
        rospy.loginfo(f"[DeepLab] mIoU: {miou_dlab:.3f}, Acc: {acc_dlab:.3f}, ClassAcc: {class_acc_dlab:.3f}")
        rospy.loginfo(f"[Pseudo]   mIoU: {miou_pseudo:.3f}, Acc: {acc_pseudo:.3f}, ClassAcc: {class_acc_pseudo:.3f}")
        rospy.loginfo(f"[SAM]      mIoU: {miou_sam:.3f}, Acc: {acc_sam:.3f}, ClassAcc: {class_acc_sam:.3f}")
        rospy.loginfo(f"[SAM Avg Changed Pixels] {sam_avg:.2f}%")
        # Publish mIoU for pseudo (you can also publish SAM if needed)
        self.miou_pub.publish(Float64(data=miou_pseudo))

        # Step 5: Save results with timestamp and scene
        result_file = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/IO_pipeline/results.txt"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write("Timestamp, Scene, Type, mIoU, Accuracy, Class Accuracy\n")

        with open(result_file, "a") as f:
            f.write(f"{current_time}, Scene {self.scene_number}, DeepLab, {miou_dlab:.3f}, {acc_dlab:.3f}, {class_acc_dlab:.3f}\n")
            f.write(f"{current_time}, Scene {self.scene_number}, Pseudo, {miou_pseudo:.3f}, {acc_pseudo:.3f}, {class_acc_pseudo:.3f}\n")
            f.write(f"{current_time}, Scene {self.scene_number}, SAM, {miou_sam:.3f}, {acc_sam:.3f}, {class_acc_sam:.3f}\n")
        rospy.loginfo("Scene execution complete, shutting down node.")
        rospy.signal_shutdown("Done")
        sys.exit(0)

if __name__ == "__main__":
    try:
        MockedControlNode(auto_yes=False).run()
    except rospy.ROSInterruptException:
        pass
