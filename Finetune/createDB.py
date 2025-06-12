import os
from pathlib import Path
from PIL import Image
import zipfile
import numpy as np

# Hyperparameters
BASE_PATH = Path("/media/adaptation/New_volume/Domain_Adaptation_Pipeline/IO_pipeline/Scannet_DB/scans")
SCENES = [f"scene{i:04d}_00" for i in range(10)]
FOLDER_NAMES = {
    "rgb": "color",
    "pseudo": "pseudo_labels_0.05",
    "sam": "sam_labels_0.05_prompt_1296x968",
    "gt": "label_nyu40"
}
OUTPUT_PATH = Path("processed_dataset")
OUTPUT_ZIP = "0.05_prompt_1296x968.zip"
TARGET_SIZE = (320, 240)
MAPPING_FILE = Path("nyu40_segmentation_mapping.csv")

# Load class color mapping
mapping = np.genfromtxt(MAPPING_FILE, delimiter=",")[1:, 1:4]
CLASS_COLORS = mapping.astype(np.uint8)


def is_grayscale(image: Image.Image) -> bool:
    return image.mode == 'L'


def convert_rgb_to_class_index(image: Image.Image) -> Image.Image:
    rgb_np = np.array(image.convert("RGB"))
    h, w = rgb_np.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        mask = np.all(np.abs(rgb_np - color) <= 5, axis=-1)
        class_map[mask] = class_idx

    return Image.fromarray(class_map, mode='L')


def process_and_save_image(image_path: Path, output_path: Path, to_index=False, is_rgb=False):
    with Image.open(image_path) as img:
        # Choose resampling method
        if img.size != TARGET_SIZE:
            if is_rgb:
                resample_mode = Image.Resampling.BOX  # Similar to cv2.INTER_AREA
            else:
                resample_mode = Image.Resampling.NEAREST
            img = img.resize(TARGET_SIZE, resample=resample_mode)

        if to_index:
            if not is_grayscale(img):
                img = convert_rgb_to_class_index(img)
            else:
                img = img.convert("L")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


def main():
    # Clean output folder if exists
    if OUTPUT_PATH.exists():
        import shutil
        shutil.rmtree(OUTPUT_PATH)
    OUTPUT_PATH.mkdir(parents=True)

    for scene in SCENES:
        for key, folder_name in FOLDER_NAMES.items():
            input_folder = BASE_PATH / scene / folder_name
            output_folder = OUTPUT_PATH / scene / key  # force standardized output folder names

            if not input_folder.exists():
                print(f"Warning: Folder {input_folder} does not exist. Skipping.")
                continue

            for img_file in input_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    to_index = key in ["pseudo", "sam", "gt"]
                    is_rgb = key == "rgb"
                    output_file = output_folder / img_file.name
                    process_and_save_image(img_file, output_file, to_index, is_rgb)

    # Zip the output folder
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_PATH):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(OUTPUT_PATH.parent))

    print(f"Processing complete. Zipped dataset saved as {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
