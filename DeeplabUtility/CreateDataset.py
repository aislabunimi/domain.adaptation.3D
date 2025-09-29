import os
import shutil
from sklearn.model_selection import train_test_split

src_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/"
scenes = ["scene0000_00", "scene0000_861"]

dst_dir = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/Training/"

for split in ["train", "val"]:
    for m in ["RGB", "GT"]:
        os.makedirs(os.path.join(dst_dir, m, split), exist_ok=True)

for scene in scenes:
    rgb_path = os.path.join(src_dir, scene, "RGB")
    gt_path = os.path.join(src_dir, scene, "GT")

    rgb_files = sorted(os.listdir(rgb_path), key=lambda x: int(os.path.splitext(x)[0]))
    basenames = [os.path.splitext(f)[0] for f in rgb_files]

    # split 80/20 una sola volta
    train_names, val_names = train_test_split(basenames, test_size=0.2, random_state=42)

    for name in train_names:
        # RGB (.jpg)
        src_rgb = os.path.join(rgb_path, name + ".jpg")
        dst_rgb = os.path.join(dst_dir, "RGB", "train", f"{scene}_{name}.jpg")
        shutil.copy(src_rgb, dst_rgb)

        # GT (.png)
        src_gt = os.path.join(gt_path, name + ".png")
        dst_gt = os.path.join(dst_dir, "GT", "train", f"{scene}_{name}.png")
        shutil.copy(src_gt, dst_gt)

    for name in val_names:
        # RGB (.jpg)
        src_rgb = os.path.join(rgb_path, name + ".jpg")
        dst_rgb = os.path.join(dst_dir, "RGB", "val", f"{scene}_{name}.jpg")
        shutil.copy(src_rgb, dst_rgb)

        # GT (.png)
        src_gt = os.path.join(gt_path, name + ".png")
        dst_gt = os.path.join(dst_dir, "GT", "val", f"{scene}_{name}.png")
        shutil.copy(src_gt, dst_gt)

print("Fatto")
