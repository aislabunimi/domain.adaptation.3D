import os
import sys
import cv2
import numpy as np
from LabelElaborator import LabelElaborator


class ColorDir:
    def __init__(self, input_dir, mapping_file=None):
        self.input_dir = input_dir
        self.output_dir = input_dir.rstrip('/\\') + "_colored"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if mapping_file is None:
            # Use nyu40_segmentation_mapping.csv in the same folder as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(script_dir, "nyu40_segmentation_mapping.csv")
            print(f"[INFO] No mapping file provided. Using default: {mapping_file}")

        if not os.path.isfile(mapping_file):
            raise FileNotFoundError(f"[ERROR] Mapping CSV not found: {mapping_file}")

        # Load RGB values from nyu40_segmentation_mapping.csv (skip header, use cols 2-4)
        mapping = np.genfromtxt(mapping_file, delimiter=",")[1:, 1:4]
        self.class_colors = mapping.astype(np.uint8)
        self.label_elaborator = LabelElaborator(self.class_colors, confidence=0)

    def colorize_directory(self):
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image_path = os.path.join(self.input_dir, filename)
                sem_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if sem_image is None:
                    print(f"[WARN] Skipping unreadable file: {filename}")
                    continue

                _, colored_sem, _ = self.label_elaborator.process(sem_image)
                output_path = os.path.join(self.output_dir, filename)
                colored_sem_bgr = cv2.cvtColor(colored_sem, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, colored_sem_bgr)
                print(f"[INFO] Saved: {output_path}")

        print(f"[DONE] Colorized images saved to: {self.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python color_dir.py <path_to_directory> [optional_path_to_mapping_csv]")
        sys.exit(1)

    input_dir = sys.argv[1]
    mapping_file = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Invalid input directory: {input_dir}")
        sys.exit(1)

    colorizer = ColorDir(input_dir, mapping_file)
    colorizer.colorize_directory()
