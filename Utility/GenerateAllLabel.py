import rospy
from sensor_msgs.msg import Image
import cv2
import os
import argparse
from TestScripts.Utilitity.PILBridge import PILBridge
from TestScripts.GenerateColoredLabel import LabelElaborator
import numpy as np
from functools import partial
from tqdm import tqdm
import time

class SceneProcessor:
    def __init__(self, base_path="/media/adaptation/New_volume/Domain_Adaptation_Pipeline"):
        self.base_path = base_path
        self.mapping_path = os.path.join(base_path, "domain.adaptation.3D/catkin_ws/src/control_node/cfg/nyu40_segmentation_mapping.csv")
        self.pending_frames = {}
        self.current_scene = None
        self.progress_bar = None
        self.label_elaborator = None
        self.deeplab_pub = None
        
    def init_ros(self):
        rospy.init_node("deeplab_sync_node", anonymous=True)
        self.deeplab_pub = rospy.Publisher('/deeplab/rgb', Image, queue_size=10)
        mapping = np.genfromtxt(self.mapping_path, delimiter=",")[1:, 1:4]
        self.label_elaborator = LabelElaborator(mapping, confidence=0)
        rospy.Subscriber("/deeplab/segmented_image", Image, self.deeplab_labels_callback)
        time.sleep(1.0)  # Allow publishers/subscribers to initialize

    def deeplab_labels_callback(self, msg):
        if self.current_scene is None or self.progress_bar is None:
            return

        try:
            sem_label = PILBridge.rosimg_to_numpy(msg)
            ts = msg.header.stamp.to_sec()

            if ts in self.pending_frames:
                filename, scene_num = self.pending_frames.pop(ts)
                scene_str = f"{scene_num:04d}"
                
                # Prepare output directories
                output_paths = {
                    'labels': os.path.join(self.base_path, f"IO_pipeline/Scannet_DB/scans/scene{scene_str}_00/deeplab_labels"),
                    'colored': os.path.join(self.base_path, f"IO_pipeline/Scannet_DB/scans/scene{scene_str}_00/deeplab_labels_colored")
                }

                for path in output_paths.values():
                    os.makedirs(path, exist_ok=True)

                # Save raw segmentation
                out_path = os.path.join(output_paths['labels'], os.path.splitext(filename)[0] + ".png")
                cv2.imwrite(out_path, sem_label)

                # Process and save colored segmentation
                _, colored_sem, _ = self.label_elaborator.process(sem_label)
                colored_sem_bgr = cv2.cvtColor(colored_sem, cv2.COLOR_RGB2BGR)
                out_path2 = os.path.join(output_paths['colored'], os.path.splitext(filename)[0] + ".png")
                cv2.imwrite(out_path2, colored_sem_bgr)

                self.progress_bar.update(1)

        except Exception as e:
            rospy.logerr_once(f"Callback error: {str(e)}")

    def process_scene(self, scene_num):
        scene_str = f"{scene_num:04d}"
        self.current_scene = scene_num
        self.pending_frames = {}  # Clear previous scene's pending frames

        input_dir = os.path.join(self.base_path, f"IO_pipeline/Scannet_DB/scans/scene{scene_str}_00/color")
        
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return False

        img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")],key=lambda x: int(x.split('.')[0]))
        
        if not img_files:
            print(f"No JPG images found in {input_dir}")
            return False

        # Initialize progress bar
        self.progress_bar = tqdm(total=len(img_files), desc=f"Processing scene {scene_str}", unit="img")

        # Process all images
        for fname in img_files:
            if rospy.is_shutdown():
                break
                
            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                self.progress_bar.write(f"Warning: Invalid image {fname}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msg = PILBridge.numpy_to_rosimg(rgb, encoding='rgb8')
            ts = rospy.Time.now()
            msg.header.stamp = ts
            self.pending_frames[ts.to_sec()] = (fname, scene_num)  # Store both filename and scene number
            self.deeplab_pub.publish(msg)
            time.sleep(0.05)  # Small throttle to avoid overwhelming the system

        # Wait for remaining callbacks
        start_time = time.time()
        while len(self.pending_frames) > 0 and not rospy.is_shutdown():
            if time.time() - start_time > 30:  # 30 second timeout
                self.progress_bar.write(f"Timeout waiting for {len(self.pending_frames)} remaining frames")
                break
            time.sleep(0.1)

        self.progress_bar.close()
        return True

def main():
    parser = argparse.ArgumentParser(description='Process ScanNet scenes')
    parser.add_argument('--scenes', type=str, required=True, help='Scene numbers (comma-separated or range, e.g. "1,2,5" or "1-5")')
    parser.add_argument('--base_path', type=str, default="/media/adaptation/New_volume/Domain_Adaptation_Pipeline", help='Base path for data directories')
    args = parser.parse_args()

    # Parse scene numbers
    scene_numbers = []
    if '-' in args.scenes:
        start, end = map(int, args.scenes.split('-'))
        scene_numbers = range(start, end+1)
    else:
        scene_numbers = list(map(int, args.scenes.split(',')))

    processor = SceneProcessor(args.base_path)
    processor.init_ros()
    
    for scene_num in scene_numbers:
        print(f"\n{'='*50}")
        print(f"Starting processing for scene {scene_num:04d}")
        print(f"{'='*50}")
        
        success = processor.process_scene(scene_num)
        
        if not success:
            print(f"Failed to process scene {scene_num:04d}")
        else:
            print(f"Successfully processed scene {scene_num:04d}")

    rospy.signal_shutdown("All scenes processed")

if __name__ == "__main__":
    main()