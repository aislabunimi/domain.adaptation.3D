import os
from pathlib import Path
from PIL import Image
import zipfile
import numpy as np

# Hyperparameters
BASE_PATH = Path("/path/to/base")  # base path containing scene folders
SCENES = [f"{6 + i:04d}_00" for i in range(10)]  # e.g., 0006_00 to 0015_00
FOLDER_NAMES = {
    "rgb": "rgb_folder",
    "pseudo": "pseudo_folder",
    "sam": "sam_folder",
    "gt": "gt_folder"
}
OUTPUT_PATH = Path("processed_dataset")
OUTPUT_ZIP = "processed_dataset.zip"
TARGET_SIZE = (320, 240)
MAPPING_FILE = Path("/path/to/mapping.csv")

# Load class color mapping
mapping = np.genfromtxt(MAPPING_FILE, delimiter=",")[1:, 1:4]
CLASS_COLORS = mapping.astype(np.uint8)


def is_grayscale(image: Image.Image) -> bool:
    """
    Check if a PIL image is grayscale.
    """
    return image.mode == 'L'


def convert_rgb_to_class_index(image: Image.Image) -> Image.Image:
    """
    Convert a PIL RGB image to class index map and return as grayscale PIL image.
    """
    rgb_np = np.array(image.convert("RGB"))
    h, w = rgb_np.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        mask = np.all(np.abs(rgb_np - color) <= 5, axis=-1)
        class_map[mask] = class_idx

    return Image.fromarray(class_map, mode='L')


def process_and_save_image(image_path: Path, output_path: Path, to_index=False):
    with Image.open(image_path) as img:
        # Resize if needed
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.ANTIALIAS)

        if to_index:
            if not is_grayscale(img):
                img = convert_rgb_to_class_index(img)
            else:
                img = img.convert("L")  # Ensure saved as 8-bit grayscale

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
            output_folder = OUTPUT_PATH / scene / folder_name

            if not input_folder.exists():
                print(f"Warning: Folder {input_folder} does not exist. Skipping.")
                continue

            for img_file in input_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    # Only pseudo, sam, gt folders should be converted to class-index grayscale
                    to_index = key in ["pseudo", "sam", "gt"]
                    output_file = output_folder / img_file.name
                    process_and_save_image(img_file, output_file, to_index)

    # Zip the output folder
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_PATH):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(OUTPUT_PATH.parent))

    print(f"Processing complete. Zipped dataset saved as {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
