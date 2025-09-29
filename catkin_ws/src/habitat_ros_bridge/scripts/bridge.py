#!/usr/bin/env python3

import habitat_sim
import rospy
import csv
import re
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from helper.PILBridge import PILBridge
from habitat_ros_bridge.msg import Sensors
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from helper.resources import NYU40_MAPPING_FILE, HABITAT_CATEGORY_MAPPING_FILE

#test_scene = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/HM3D/minival/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb"
#test_scene = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/HM3D/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
test_scene = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/HM3D/minival/00808-y9hTuugGdiq/y9hTuugGdiq.glb"
scene_config = "/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/HM3D/minival/hm3d_annotated_basis.scene_dataset_config.json"

settings = {
    "scene": test_scene,            # Scene path
    "scene_conf": scene_config,
    "default_agent": 0,             # Index of the default agent
    "sensor_height": 1,           # Height of sensors in meters, relative to the agent
    "width": 640,                   # Image width
    "height": 480,                  # Image height
}

def setup_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_conf"]

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

    sem_spec = habitat_sim.CameraSensorSpec()
    sem_spec.uuid = "semantic_sensor"
    sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_spec.resolution = [settings["height"], settings["width"]]
    sem_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, sem_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

def category_to_nyu40_dict():

    tsv_file = os.path.normpath(HABITAT_CATEGORY_MAPPING_FILE)

    category_to_nyu40 = {}

    with open(tsv_file, encoding='utf-8') as f:

        header_line = f.readline().strip()
        header = re.split(r' {2,}', header_line)  # split su 2 o più spazi
        cat_idx = header.index('category')
        nyu40_idx = header.index('nyu40id')
        
        for line in f:
            line = line.strip()
            parts = re.split(r' {2,}', line)  # split su 2 o più spazi
            category_name = parts[cat_idx].strip()
            nyu40_id = int(parts[nyu40_idx].strip())
            category_to_nyu40[category_name] = nyu40_id

    return category_to_nyu40

def instance_id_to_category_dict(scene):

    # obj.id is an instace id, each object is named with the category and a global incremental number
    # light fixture_6 
    # light fixture_7
    # wall_8

    # category.index() is an index of the category given in order
    # light fixture_6 2
    # light fixture_7 2
    # wall_8 3
    # lamp_9 4
    # wall_10 3

    instance_id_to_category = {
        int(obj.id.split("_")[-1]): obj.category.name() 
        for obj in scene.objects
    }

    return instance_id_to_category

def instance_to_nyu40(obs, instance_id_to_category, category_to_nyu40):
    
    nyu40_labels = np.zeros_like(obs, dtype=np.uint8)

    for instance_id, category_name in instance_id_to_category.items():
        if category_name in category_to_nyu40:
            nyu40_id = category_to_nyu40[category_name]
        else:
            nyu40_id = 0  # "unlabeled"
        nyu40_labels[obs == instance_id] = nyu40_id

    return nyu40_labels


class HabitatROSBridge:
    def __init__(self):

        rospy.init_node('habitat_ros_bridge', anonymous=True)
        self.sim = setup_sim()

        self.agent = self.sim.get_agent(0)
        sensor_spec = self.sim.get_agent(0).agent_config.sensor_specifications[1]
        width, height = sensor_spec.resolution[1], sensor_spec.resolution[0]

        hfov_rad = float(sensor_spec.hfov) * np.pi / 180.0
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))

        fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
        fy = (height / 2.0) / np.tan(vfov_rad / 2.0)

        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0., cx, 0.],
            [0., fy, cy, 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]
        ])
        #np.savetxt("/media/adaptation/New_volume/Domain_Adaptation_Pipeline/ColomboHM3D/DataSet/scene0000_08/INTRINSIC/intrinsic_depth.txt", K, fmt="%.6f")
        

        agent_state = habitat_sim.AgentState()
        # [1.0, 3.0, 1.0]
        #2 altezza
        agent_state.position = np.array([0.0, -1.8, 0.0], dtype=np.float32)
        agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identità, guarda dritto
        self.agent.set_state(agent_state)

        self.category_to_nyu40 = category_to_nyu40_dict()

        scene = self.sim.semantic_scene
        self.instance_id_to_category = instance_id_to_category_dict(scene)

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
            action = "stop"
        self.agent.act(action)

    def publish_images(self):

        obs = self.sim.get_sensor_observations()
        
        rgb = obs["color_sensor"][:, :, :3]
        rgb_msg = PILBridge.numpy_to_rosimg(rgb, frame_id="habitat_rgb_camera",encoding="rgb8")
        
        depth = obs["depth_sensor"]
        depth_32fc1 = depth.astype(np.float32)
        depth_msg = PILBridge.numpy_to_rosimg(depth_32fc1,frame_id="habitat_depth_camera",encoding="32FC1")

        sem_nyu40 = instance_to_nyu40(obs["semantic_sensor"], self.instance_id_to_category, self.category_to_nyu40)
        semantic_msg = PILBridge.numpy_to_rosimg(sem_nyu40, frame_id="habitat_semantic_camera", encoding="mono8")

        #T = self.sim.get_agent(0)._sensors["color_sensor"].node.absolute_transformation()
        # Converti in numpy (float32 o float64 a scelta)
        #T_np = np.array(T, dtype=np.float64).reshape(-1)  # shape (4, 4)
        
        agent_position = self.sim.get_agent(0).get_state().position
        agent_rotation = self.sim.get_agent(0).get_state().rotation # [w, x, y, z]

        w, x, y, z = agent_rotation.w, agent_rotation.x, agent_rotation.y, agent_rotation.z

        # Riordina in [x, y, z, w] per SciPy e converti in float64
        quat_np = np.array([x, y, z, w], dtype=np.float64)

        # Converti quaternion in matrice di rotazione 3x3
        # Crea un oggetto Rotation da SciPy
        rotation = R.from_quat(quat_np)

        # Ottieni la matrice di rotazione 3x3
        R_matrix = rotation.as_matrix()

        # Costruisci la matrice di trasformazione 4x4
        agent_pose = np.eye(4)
        agent_pose[:3, :3] = R_matrix
        agent_pose[:3, 3] = agent_position

        pose_reshaped = agent_pose.reshape(-1)
    
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