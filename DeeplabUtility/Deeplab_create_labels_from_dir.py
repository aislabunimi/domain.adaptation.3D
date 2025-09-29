#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import os
import numpy as np
from helper.PILBridge import PILBridge
from helper.LabelElaborator import LabelElaborator
import time
from tqdm import tqdm
from helper.resources import NYU40_MAPPING_FILE

#INPUT_DIR = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/ReplicaDataset/room_0/RGB/"
INPUT_DIR = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene00808_1/RGB/"


OUTPUT_DIRS = {
    "labels": os.path.join(os.path.dirname(INPUT_DIR.rstrip("/")), "deeplab"),
    "colored": os.path.join(os.path.dirname(INPUT_DIR.rstrip("/")), "deeplab_colored"),
}

def empty_folder(path):
    """Delete all files in a folder (keeps the folder)."""
    if os.path.isdir(path):
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        os.makedirs(path)



class DirProcessor:

    def __init__(self):

        self.pending_frames = {}
        self.progress_bar = None
        self.label_elaborator = None
        self.deeplab_pub = None

        for d in OUTPUT_DIRS.values():
            os.makedirs(d, exist_ok=True)
            empty_folder(d)

    def init_ros(self):

        rospy.init_node("deeplab_dir_node", anonymous=True)

        self.deeplab_pub = rospy.Publisher("/deeplab/rgb", Image, queue_size=10)

        mapping = np.genfromtxt(NYU40_MAPPING_FILE, delimiter=",")[1:, 1:4]
        self.label_elaborator = LabelElaborator(mapping, confidence=0)

        rospy.Subscriber("/deeplab/segmented_image", Image, self.seg_callback)
        rospy.sleep(1.0)

    def seg_callback(self, msg):
        if self.progress_bar is None:
            return

        try:
            sem_label = PILBridge.rosimg_to_numpy(msg)
            ts = msg.header.stamp.to_sec()

            if ts in self.pending_frames:
                filename = self.pending_frames.pop(ts)

                out_path = os.path.join(OUTPUT_DIRS["labels"], os.path.splitext(filename)[0] + ".png")
                cv2.imwrite(out_path, sem_label)

                _, colored_sem, _ = self.label_elaborator.process(sem_label)
                colored_sem_bgr = cv2.cvtColor(colored_sem, cv2.COLOR_RGB2BGR)
                out_path2 = os.path.join(OUTPUT_DIRS["colored"], os.path.splitext(filename)[0] + ".png")
                cv2.imwrite(out_path2, colored_sem_bgr)

                self.progress_bar.update(1)
        except Exception as e:
            rospy.logerr_once(f"Callback error: {str(e)}")

    def process_dir(self):
        img_files = sorted(
            [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        if not img_files:
            print(f"No JPG images found in {INPUT_DIR}")
            return False

        self.progress_bar = tqdm(total=len(img_files), desc="Processing directory", unit="img")

        for fname in img_files:
            if rospy.is_shutdown():
                break

            img_path = os.path.join(INPUT_DIR, fname)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.progress_bar.write(f"Warning: invalid image {fname}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msg = PILBridge.numpy_to_rosimg(rgb, encoding="rgb8")
            ts = rospy.Time.now()
            msg.header.stamp = ts
            self.pending_frames[ts.to_sec()] = fname
            self.deeplab_pub.publish(msg)
            time.sleep(0.1)

        # Wait for callbacks to finish
        start_time = time.time()
        while len(self.pending_frames) > 0 and not rospy.is_shutdown():
            if time.time() - start_time > 30:
                self.progress_bar.write(f"Timeout waiting for {len(self.pending_frames)} remaining frames")
                break
            time.sleep(0.1)

        self.progress_bar.close()
        return True


def main():
    processor = DirProcessor()
    processor.init_ros()

    print(f"Starting processing for directory: {INPUT_DIR}")
    success = processor.process_dir()

    if success:
        print(f"Successfully processed {INPUT_DIR}")
    else:
        print(f"Failed to process {INPUT_DIR}")

    rospy.signal_shutdown("Directory processed")


if __name__ == "__main__":
    main()
