#!/usr/bin/env python3

import habitat_sim
import rospy
import numpy as np
from helper.PILBridge import PILBridge
from habitat_ros_bridge.msg import Sensors
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import json

# Path to Replica semantic metadata (json, not ply!)
json_labels = "/media/adaptation/D435-A0D8/Replica_dataset/Dataset/room_0/habitat/info_semantic.json"
test_scene = "/media/adaptation/D435-A0D8/Replica_dataset/Dataset/room_0/habitat/mesh_semantic.ply"

settings = {
    "scene": test_scene,
    "default_agent": 0,
    "sensor_height": 1.0,
    "width": 640,
    "height": 480,
}

""" 
to realize the semnatic mapping from instance to nyu40 i have to
take the instance for each obj 
for each instance take the instance id
using the json contained in each scene convert the instance id to name
Using the mapping that i made convert the name to nyu40 label
"""

# Mapping Replica category name -> NYU40 id
replicaName_to_nyu40 = {
    "book": 23, "wall": 1, "lamp": 35, "wall-plug": 40, "blinds": 13, "chair": 5,
    "table": 7, "door": 8, "cushion": 18, "bowl": 40, "window": 9, "switch": 40,
    "anonymize_text": 40, "bottle": 40, "anonymize_picture": 11, "indoor-plant": 40,
    "cup": 40, "box": 29, "vent": 38, "ceiling": 22, "pillow": 18, "panel": 38,
    "vase": 40, "handrail": 38, "plate": 40, "floor": 2, "clothing": 21, "pot": 40,
    "basket": 40, "non-plane": 38, "rug": 20, "shoe": 40, "plant-stand": 39,
    "pillar": 38, "cabinet": 3, "bin": 40, "rack": 39, "camera": 40, "shelf": 15,
    "tv-screen": 25, "picture": 11, "blanket": 27, "sofa": 6, "nightstand": 32,
    "small-appliance": 40, "mat": 20, "countertop": 12, "beanbag": 6, "bike": 40,
    "sink": 34, "base-cabinet": 3, "faucet": 34, "kitchen-utensil": 40,
    "wall-cabinet": 3, "tissue-paper": 26, "chopping-board": 40, "curtain": 16,
    "tablet": 40, "major-appliance": 24, "remote-control": 40, "scarf": 21,
    "sculpture": 40, "bed": 4, "stair": 38, "tv-stand": 39, "refrigerator": 24,
    "stool": 5, "comforter": 27, "umbrella": 40, "plane": 38, "knife-block": 40,
    "handbag": 37, "pan": 40, "clock": 40, "shower-stall": 28, "towel": 27,
    "toilet": 33, "desk": 14, "pipe": 38, "bench": 39, "cloth": 21, "candle": 40,
    "desk-organizer": 40, "utensil-holder": 40, "coaster": 40, "bathtub": 36,
    "cooktop": 12, "monitor": 25
}

def setup_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    sem_sensor_spec = habitat_sim.CameraSensorSpec()
    sem_sensor_spec.uuid = "semantic_sensor"
    sem_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_sensor_spec.resolution = [settings["height"], settings["width"]]
    sem_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, sem_sensor_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

def build_instance_to_replicaId(scene):
    """
    Returns a dict: instance_id -> nyu40_label
    """
    mapping = {}
    for obj in scene.objects:
        try:
            instance_id = int(obj.id.split("_")[-1])  # e.g. "chair_12"
        except:
            continue
        if obj.category is not None and obj.category.index() is not None:
            mapping[instance_id] = obj.category.index()
        else:
            mapping[instance_id] = 0  # unlabeled
    return mapping


def build_replicaId_to_nyu40(json_file):
    """
    Build dict: instance_id -> NYU40 id
    """
    with open(json_file, "r") as f:
        semantic_info = json.load(f)

    instance_to_nyu40 = {}
    for obj in semantic_info["classes"]:
        instance_id = obj["id"]
        name = obj["name"]
        nyu40_id = replicaName_to_nyu40.get(name, 0)  # 0 = unlabeled
        instance_to_nyu40[instance_id] = nyu40_id

    return instance_to_nyu40


def instance_to_nyu40_image(obs, instance_to_replicaID, replicaID_to_nyu40):
    nyu40_img = np.zeros_like(obs, dtype=np.uint8)

    for instance_id in np.unique(obs):
        if instance_id in instance_to_replicaID:
            replica_id = instance_to_replicaID[instance_id]
            nyu40_id = replicaID_to_nyu40.get(replica_id, 0)  # 0 = unlabeled
        else:
            nyu40_id = 0

        nyu40_img[obs == instance_id] = nyu40_id

    return nyu40_img


class HabitatROSBridge:
    def __init__(self):
        rospy.init_node("habitat_ros_bridge", anonymous=True)
        self.sim = setup_sim()
        self.agent = self.sim.get_agent(0)

        # Camera intrinsics
        sensor_spec = self.agent.agent_config.sensor_specifications[0]
        width, height = sensor_spec.resolution[1], sensor_spec.resolution[0]
        hfov_rad = float(sensor_spec.hfov) * np.pi / 180.0
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))

        fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
        fy = (height / 2.0) / np.tan(vfov_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0

        K = np.array([
            [fx, 0., cx, 0.],
            [0., fy, cy, 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]
        ])
        #np.savetxt("intrinsic_depth.txt", K, fmt="%.6f")


        #agent_state = habitat_sim.AgentState()
        # [1.0, 3.0, 1.0] robot height is default 1.5
        #2 altezza
        #agent_state.position = np.array([1.0, -0.8, 0.0], dtype=np.float32)
        #agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identità, guarda dritto
        #self.agent.set_state(agent_state)

        scene = self.sim.semantic_scene
        # Build mapping from instance ID -> NYU40 ID
        self.instance_to_replicaID = build_instance_to_replicaId(scene)
        self.replicaID_to_nyu40 = build_replicaId_to_nyu40(json_labels)


        rospy.Subscriber("/cmd_vel", Twist, self.cmd_callback)
        self.scene_sensors_pub = rospy.Publisher("/habitat/scene/sensors", Sensors, queue_size=10)

        self.run()

    def cmd_callback(self, msg):
        if msg.linear.x > 0:
            action = "move_forward"
        elif msg.linear.x < 0:
            action = "move_backward"
        elif msg.linear.y > 0:
            action = "move_up"
        elif msg.linear.y < 0:
            action = "move_down"
        elif msg.angular.z > 0:
            action = "turn_left"
        elif msg.angular.z < 0:
            action = "turn_right"
        else:
            action = None
        if action is not None:
            self.agent.act(action)

    def publish_images(self):
        obs = self.sim.get_sensor_observations()

        # RGB
        rgb = obs["color_sensor"][:, :, :3]
        rgb_msg = PILBridge.numpy_to_rosimg(rgb, frame_id="habitat_rgb_camera", encoding="rgb8")

        # Depth
        depth = obs["depth_sensor"]
        depth_msg = PILBridge.numpy_to_rosimg(depth.astype(np.float32),
                                              frame_id="habitat_depth_camera",
                                              encoding="32FC1")

        # Semantic (NYU40 labels)
        sem_obs = obs["semantic_sensor"]
        sem_nyu40 = instance_to_nyu40_image(sem_obs, self.instance_to_replicaID, self.replicaID_to_nyu40)
        semantic_msg = PILBridge.numpy_to_rosimg(sem_nyu40.astype(np.uint8),
                                                 frame_id="habitat_semantic_camera",
                                                 encoding="mono8")

        # Pose
        agent_state = self.agent.get_state()
        pos = agent_state.position
        rot = agent_state.rotation  # quaternion [w, x, y, z]

        quat_np = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64)
        rotation = R.from_quat(quat_np)
        R_matrix = rotation.as_matrix()

        agent_pose = np.eye(4)
        agent_pose[:3, :3] = R_matrix
        agent_pose[:3, 3] = pos
        pose_reshaped = agent_pose.reshape(-1)

        # Publish
        sensors_msg = Sensors()
        sensors_msg.rgb = rgb_msg
        sensors_msg.depth = depth_msg
        sensors_msg.sem = semantic_msg
        sensors_msg.pose = pose_reshaped
        self.scene_sensors_pub.publish(sensors_msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_images()
            rate.sleep()

if __name__ == "__main__":
    try:
        HabitatROSBridge()
    except rospy.ROSInterruptException:
        pass
