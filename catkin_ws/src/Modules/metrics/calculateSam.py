import numpy as np
import os
from tqdm import tqdm
import cv2
import rospy
from metrics.SamMetrics import SamMetrics
import matplotlib.pyplot as plt
import seaborn as sns
from helper.resources import NYU40_MAPPING_FILE

def rgb_to_class_index(rgb_image, tolerance=5):

    mapping = np.genfromtxt(NYU40_MAPPING_FILE, delimiter=",")[1:, 1:4]

    h, w = rgb_image.shape[:2]
    class_map = np.zeros((h, w), dtype=np.int32)
    for class_idx, color in enumerate(mapping):
        mask = np.all(np.abs(rgb_image - color) <= tolerance, axis=-1)
        class_map[mask] = class_idx
    return class_map

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
    
def calculate_sam_metrics(pred_dir, pseudo_dir,gt_dir, scene_number, log_path="sam_metrics_log.json", resize_to=(320, 240), file_ext=".png"):
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
        pseudo_img = rgb_to_class_index(pseudo_img)

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

        sam_meter.update(frame_id=f, info=scene_number, original=pseudo_img, gt=gt_img, refined=pred_img)

    median_change, global_conf_matrix = sam_meter.measure()
    sam_meter.save_log()
    return median_change, global_conf_matrix