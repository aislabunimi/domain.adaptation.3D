import os
import rospy
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm
from helper.resources import NYU40_MAPPING_FILE

def rgb_to_class_index(rgb_image, tolerance=5):

    mapping = np.genfromtxt(NYU40_MAPPING_FILE, delimiter=",")[1:, 1:4]

    h, w = rgb_image.shape[:2]
    class_map = np.zeros((h, w), dtype=np.int32)
    for class_idx, color in enumerate(mapping):
        mask = np.all(np.abs(rgb_image - color) <= tolerance, axis=-1)
        class_map[mask] = class_idx
    return class_map

def calculate_metrics(pred_dir, gt_dir, meter, resize_to=(320, 240), file_ext=".png", perc=1):
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

    meter.clear()
    
    missing_classes_counter = Counter()
    missing_class_counts = []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(file_ext)], key=lambda x: int(x.split('.')[0]))
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
            rospy.loginfo_once(f"[INFO] Resizing prediction {f} from {pred_img.shape[::-1]} to {resize_to}")
            pred_img = cv2.resize(pred_img, resize_to, interpolation=cv2.INTER_NEAREST)

        # Resize ground truth if needed
        if gt_img.shape[:2] != resize_to[::-1]:
            rospy.loginfo_once(f"[INFO] Resizing ground truth {f} from {gt_img.shape[::-1]} to {resize_to}")
            gt_img = cv2.resize(gt_img, resize_to, interpolation=cv2.INTER_NEAREST)

        # Convert to int and shift class IDs if needed
        if len(pred_img.shape) == 3 and pred_img.shape[2] == 3:
            
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            pred_img = rgb_to_class_index(pred_img)
        else:
            rospy.logwarn_once("Prediction image is grayscale, skipping rgb_to_class_index conversion.")

        """
        # Convert to int and shift class IDs if needed, GT
        if len(gt_img.shape) == 3 and gt_img.shape[2] == 3:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = rgb_to_class_index(gt_img, )
        else:
            rospy.logwarn_once("Prediction image is grayscale, skipping rgb_to_class_index conversion.")
        """
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