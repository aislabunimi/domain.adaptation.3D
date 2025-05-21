import numpy as np
import cv2
import os
from tqdm import tqdm
from LabelElaborator import LabelElaborator

def main():
    scan_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/IO_pipeline/Scannet_DB/scans"
    cat_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/domain.adaptation.3D/catkin_ws/src"
    scene_dir = "/scene0000_00"

    mapping_path = os.path.join(cat_dir, "control_node/cfg/nyu40_segmentation_mapping.csv")
    mapping = np.genfromtxt(mapping_path, delimiter=",")[1:, 1:4]
    label_elaborator = LabelElaborator(mapping, confidence=0)

    input_sam_dir = os.path.join(scan_dir, scene_dir.strip("/"), "sam_labels_0.03")
    output_path = os.path.join(scan_dir, scene_dir.strip("/"), "sam_labels_0.03_colored")
    os.makedirs(output_path, exist_ok=True)

    sam_files = sorted(
        [f for f in os.listdir(input_sam_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for f in tqdm(sam_files, desc="elaborating sam"):
        input_path = os.path.join(input_sam_dir, f)

        sam_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if sam_image is None:
            print(f"Warning: could not read {input_path}")
            continue

        _, colored_sam, _ = label_elaborator.process(sam_image)

        colored_sam_bgr = cv2.cvtColor(colored_sam, cv2.COLOR_RGB2BGR)

        output_file = os.path.join(output_path, f)
        cv2.imwrite(output_file, colored_sam_bgr)

if __name__ == "__main__":
    main()

