import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# Hyperparameters
BASE_PATH = Path("/media/adaptation/New_volume/Domain_Adaptation_Pipeline/IO_pipeline/Scannet_DB/scans")
SCENES = [f"scene{i:04d}_00" for i in range(6,10)]
FOLDER_NAMES = {
    "rgb": "color",
    "pseudo3": "pseudo_labels_0.03",
    "pseudo5": "pseudo_labels_0.05",
    "gt": "label_nyu40",
    "deeplab": "deeplab_labels",
    "samAs5": "sam_labels_0.05_auto_320x240",
    "samAb5": "sam_labels_0.05_auto_1296x968",
    "samCs5": "sam_labels_0.05_prompt_320x240",
    "samCb5": "sam_labels_0.05_prompt_1296x968",
    "samAs3": "sam_labels_0.03_auto_320x240",
    "samAb3": "sam_labels_0.03_auto_1296x968",
    "samCs3": "sam_labels_0.03_prompt_320x240",
    "samCb3": "sam_labels_0.03_prompt_1296x968"
}
OUTPUT_PATH = Path("processed_dataset")
TARGET_SIZE = (320, 240)
MAPPING_FILE = Path("nyu40_segmentation_mapping.csv")

# Load class color mapping
mapping = np.genfromtxt(MAPPING_FILE, delimiter=",")[1:, 1:4]
CLASS_COLORS = mapping.astype(np.uint8)


def convert_rgb_to_class_index(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    class_map = np.zeros((h, w), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        mask = np.all(np.abs(image - color) <= 5, axis=-1)
        class_map[mask] = class_idx

    return class_map


def resize_image(img: np.ndarray, is_rgb: bool) -> np.ndarray:
    interpolation = cv2.INTER_AREA if is_rgb else cv2.INTER_NEAREST
    return cv2.resize(img, TARGET_SIZE, interpolation=interpolation)


def process_and_save_image(image_path: Path, output_path: Path, to_index=False, is_rgb=False):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    if is_rgb and img.ndim == 2:  # grayscale but expected rgb
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif not is_rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize only if needed
    if (img.shape[1], img.shape[0]) != TARGET_SIZE:
        img = resize_image(img, is_rgb)

    # Convert only if it's not already class-indexed
    if to_index:
        if img.ndim == 3 and img.shape[2] == 3:
            img = convert_rgb_to_class_index(img)
        elif img.ndim == 2 and img.dtype != np.uint8:
            img = img.astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if to_index or not is_rgb:
        cv2.imwrite(str(output_path), img)
    else:
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    # Clean output folder if exists
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(parents=True)
    

    for scene in SCENES:
        for key, folder_name in FOLDER_NAMES.items():
            input_folder = BASE_PATH / scene / folder_name
            output_folder = OUTPUT_PATH / scene / key

            if not input_folder.exists():
                print(f"Warning: Folder {input_folder} does not exist. Skipping.")
                continue

            for img_file in input_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    to_index = key not in ["rgb"]
                    is_rgb = key == "rgb"
                    output_file = output_folder / img_file.name
                    process_and_save_image(img_file, output_file, to_index, is_rgb)

    print("Processing complete. Images saved in", OUTPUT_PATH)


if __name__ == "__main__":
    main()
