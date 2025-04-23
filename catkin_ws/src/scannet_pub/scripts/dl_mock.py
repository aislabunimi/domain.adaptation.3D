#!/usr/bin/env python3

import os
import time
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader

import rospy
import rospkg
from std_msgs.msg import Empty
from sensor_msgs.msg import CameraInfo
import geometry_msgs.msg
import tf2_ros
import tf_conversions

from kimera_interfacer.msg import SyncSemantic
from habitat_ros_bridge.msg import Sensors

from Modules import PILBridge
from label_loader import LabelLoaderAuto

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore")



def txt_to_camera_info(cam_p, img_p):
    data = np.loadtxt(cam_p)
    img = imageio.imread(img_p)

    msg = CameraInfo()
    msg.width = img.shape[1]
    msg.height = img.shape[0]
    msg.K = data[:3, :3].reshape(-1).tolist()
    msg.D = [0, 0, 0, 0, 0]
    msg.R = data[:3, :3].reshape(-1).tolist()
    msg.P = data[:3, :4].reshape(-1).tolist()
    msg.distortion_model = "plumb_bob"
    return msg


def broadcast_camera_pose(H, frames, stamp):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = stamp
    t.header.frame_id = frames[0]
    t.child_frame_id = frames[1]
    t.transform.translation.x = H[0, 3]
    t.transform.translation.y = H[1, 3]
    t.transform.translation.z = H[2, 3]
    q = tf_conversions.transformations.quaternion_from_matrix(H)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    br.sendTransform(t)


class RGBDSemanticDataset(Dataset):
    def __init__(self, aux_flags, label_paths, depth_paths, image_paths, camera_info_img, camera_info_depth, sub_factor):
        self.aux = aux_flags
        self.labels = label_paths
        self.depths = depth_paths
        self.images = image_paths
        self.sub_factor = sub_factor

        self.K_img = np.array(camera_info_img.K).reshape(3, 3)
        self.K_depth = np.array(camera_info_depth.K).reshape(3, 3)

        self.map1, self.map2 = cv.initUndistortRectifyMap(
            self.K_img, np.zeros(5), np.eye(3), self.K_depth, (640, 480), cv.CV_32FC1
        )

        self.length = len(self.labels)
        for i, (l, d) in enumerate(zip(self.labels, self.depths)):
            if not os.path.isfile(l) or not os.path.isfile(d):
                self.length = i
                break

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label_p, depth_p, image_p, is_aux = self.labels[idx], self.depths[idx], self.images[idx], self.aux[idx]

        depth = imageio.imread(depth_p)
        mask = np.zeros_like(depth)
        mask[::self.sub_factor, ::self.sub_factor] = 1
        depth[mask != 1] = 0

        rgb = imageio.imread(image_p)
        sem, _ = label_loader_aux.get(label_p) if is_aux else label_loader.get(label_p)

        sem_img = np.zeros((*sem.shape, 3), dtype=np.uint8)
        for i in range(41):
            sem_img[sem == i] = color_map[i]

        rgb = cv.remap(rgb, self.map1, self.map2, interpolation=cv.INTER_NEAREST)
        sem_img = cv.remap(sem_img, self.map1, self.map2, interpolation=cv.INTER_NEAREST)

        pose_id = int(os.path.basename(label_p).split('.')[0])
        H_cam = np.loadtxt(f"{scene_dir}/pose/{pose_id}.txt")

        return rgb, depth.astype(np.int32), sem_img, H_cam


def start_cb(msg):
    global explore_active
    explore_active = True
    rospy.loginfo("Started explore stream")


def stop_cb(msg):
    global explore_active
    explore_active = False
    rospy.loginfo("Stopped explore stream")


def reset_cb(msg):
    global reset_requested
    reset_requested = True
    rospy.loginfo("Reset explore stream")


def dl_mock():
    global label_loader, label_loader_aux, color_map, scene_dir
    global streaming, stop_requested, reset_requested, explore_active
    streaming = False
    stop_requested = False
    reset_requested = False
    explore_active = False
    rospy.init_node("dl_mock", anonymous=True)
    rospy.loginfo("Launching dl_mock node...")

    # Params
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("kimera_interfacer")
    fps = rospy.get_param("~/dl_mock/fps", 5)
    base_link = rospy.get_param("~/dl_mock/base_link_frame")
    world_frame = rospy.get_param("~/dl_mock/world_frame")
    scene_dir = rospy.get_param("~/dl_mock/scannet_scene_dir")
    label_dir = rospy.get_param("~/dl_mock/label_scene_dir")
    aux_dirs = rospy.get_param("~/dl_mock/aux_labels", [])
    sub_factor = rospy.get_param("~/dl_mock/sub_reprojected", 4)
    frame_limit = rospy.get_param("~/dl_mock/frame_limit", -1)
    sync_topic = rospy.get_param("~/dl_mock/sync_topic", "/sync_semantic")

    # Loaders
    label_loader = LabelLoaderAuto(rospy.get_param("~/dl_mock/root_scannet"),
                                   rospy.get_param("~/dl_mock/prob_main"))
    label_loader_aux = LabelLoaderAuto(rospy.get_param("~/dl_mock/root_scannet"),
                                       rospy.get_param("~/dl_mock/prob_aux"))

    mapping = np.genfromtxt(f"{pkg_path}/cfg/nyu40_segmentation_mapping.csv", delimiter=",")
    color_map = mapping[1:, 1:4].astype(np.uint8)

    # Publishers
    sensor_pub = rospy.Publisher("/habitat/scene/sensors", Sensors, queue_size=1)
    #sync_pub = rospy.Publisher(sync_topic, SyncSemantic, queue_size=1)
    #image_cam_pub = rospy.Publisher("rgb_camera_info", CameraInfo, queue_size=1)
    #depth_cam_pub = rospy.Publisher("depth_camera_info", CameraInfo, queue_size=1)

    # Subscribers
    rospy.Subscriber("/startexplore", Empty, start_cb)
    rospy.Subscriber("/stopexplore", Empty, stop_cb)
    rospy.Subscriber("/resetexplore", Empty, reset_cb)

    # Camera info
    camera_info_img = txt_to_camera_info(f"{scene_dir}/intrinsic/intrinsic_color.txt", f"{scene_dir}/color/0.jpg")
    camera_info_depth = txt_to_camera_info(f"{scene_dir}/intrinsic/intrinsic_depth.txt", f"{scene_dir}/color/0.jpg")

    # File lists
    labels = sorted([str(p) for p in Path(label_dir).rglob("*.png") if "_.png" not in str(p) and int(p.name[:-4]) % 10 == 0])
    depths = [f"{scene_dir}/depth/{Path(p).name}" for p in labels]
    images = [f"{scene_dir}/color/{Path(p).stem}.jpg" for p in labels]
    aux_flags = [False] * len(labels)

    # Add auxiliary labels if present
    for aux_path in aux_dirs:
        if os.path.isdir(aux_path):
            for i, label in enumerate(labels):
                labels.append(os.path.join(aux_path, os.path.basename(label)))
                depths.append(depths[i])
                images.append(images[i])
                aux_flags.append(True)

    dataset = RGBDSemanticDataset(aux_flags, labels, depths, images, camera_info_img, camera_info_depth, sub_factor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    rate = rospy.Rate(fps)
    iterator = iter(dataloader)
    frame_idx = 0

    while not rospy.is_shutdown():
        if not explore_active:
            rate.sleep()
            continue

        if reset_requested:
            iterator = iter(dataloader)
            frame_idx = 0
            reset_requested = False
            rospy.loginfo("🌀 Resetting data stream...")

        try:
            rgb, depth, sem_img, H_cam = next(iterator)
        except StopIteration:
            rospy.loginfo("✅ Finished streaming all frames.")
            explore_active = False  # Stop streaming
            reset_requested = True  # Prepare for restart
            rospy.loginfo("🔁 Ready to restart stream on /startexplore")
            continue

        if frame_limit != -1 and frame_idx >= frame_limit:
            rospy.loginfo("📦 Reached frame limit.")
            explore_active = False
            reset_requested = True
            rospy.loginfo("🔁 Ready to restart stream on /startexplore")
            continue

        t = rospy.Time.now()

        img_msg = PILBridge.PILBridge.numpy_to_rosimg(rgb[0].numpy(), encoding="rgb8")
        depth_msg = PILBridge.PILBridge.numpy_to_rosimg(depth[0].numpy().astype(np.uint16), encoding="16UC1")
        sem_msg = PILBridge.PILBridge.numpy_to_rosimg(sem_img[0].numpy(), encoding="rgb8")

        for msg in (img_msg, depth_msg, sem_msg):
            msg.header.frame_id = "base_link_gt"
            msg.header.seq = frame_idx
            msg.header.stamp = t

        # Uncomment if you want to send semantic sync
        # sync = SyncSemantic(image=img_msg, depth=depth_msg, sem=sem_msg)
        pose_flat = H_cam[0].numpy().flatten().tolist()

        sensors = Sensors(
            rgb=img_msg,
            depth=depth_msg,
            pose=pose_flat  
        )

        # Commenting out TF broadcast since we're embedding the pose now
        # broadcast_camera_pose(H_cam[0].numpy(), (world_frame, base_link), t)

        sensor_pub.publish(sensors)

        # sync_pub.publish(sync)  # ← Commented out as requested
        #image_cam_pub.publish(camera_info_img)
        #depth_cam_pub.publish(camera_info_depth)

        rospy.loginfo(f"[{frame_idx}] 📤 Published frame at {t.to_sec():.2f}")
        frame_idx += 1
        rate.sleep()


    rospy.loginfo("Exploration mock node finished. Sleeping before shutdown.")
    time.sleep(2)


if __name__ == "__main__":
    try:
        dl_mock()
    except rospy.ROSInterruptException:
        pass
