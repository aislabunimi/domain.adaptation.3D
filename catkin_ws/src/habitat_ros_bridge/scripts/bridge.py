#!/usr/bin/env python3

import habitat_sim
import rospy
import numpy as np
from Modules import PILBridge
from habitat_ros_bridge.msg import Sensors
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

test_scene = "/home/michele/Desktop/Colombo/HM3D/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
scene_config = "/home/michele/Desktop/Colombo/HM3D/hm3d_annotated_basis.scene_dataset_config.json"

settings = {
    "scene": test_scene,            # Scene path
    "scene_conf": scene_config,
    "default_agent": 0,             # Index of the default agent
    "sensor_height": 0.7,           # Height of sensors in meters, relative to the agent
    "width": 640,                   # Image width
    "height": 480,                  # Image height
}

def setup_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_conf"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # RGB visual sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    # Semantic sensor
    sem_spec = habitat_sim.CameraSensorSpec()
    sem_spec.uuid = "semantic_sensor"
    sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_spec.resolution = [settings["height"], settings["width"]]
    sem_spec.position = [0.0, settings["sensor_height"], 0.0]

    # Add both sensors to the agent
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, sem_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


class HabitatROSBridge:
    def __init__(self):
        rospy.init_node('habitat_ros_bridge', anonymous=True)
        self.sim = setup_sim()
        self.agent = self.sim.get_agent(0)

        rospy.loginfo(self.sim)
        
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_callback)
        self.scene_sensors_pub = rospy.Publisher("/habitat/scene/sensors", Sensors, queue_size=10)
        self.semantic_pub = rospy.Publisher("/habitat/semantic", Image, queue_size=10)

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
        
        # Publish RGB and Depth by Sensors msg

        rgb = obs["color_sensor"][:, :, :3]  # Extract RGB data
        rgb_msg = PILBridge.numpy_to_rosimg(rgb,frame_id="habitat_rgb_camera",encoding="rgb8")
        
        depth = obs["depth_sensor"]
        depth_32fc1 = depth.astype(np.float32)
        depth_msg =PILBridge.numpy_to_rosimg(depth_32fc1,frame_id="habitat_depth_camera",encoding="32FC1")

        sensors_msg = Sensors()
        sensors_msg.rgb = rgb_msg
        sensors_msg.depth = depth_msg
        self.scene_sensors_pub.publish(sensors_msg)

                
        # Publish semantic image
        semantic = obs["semantic_sensor"]
        pil_img = habitat_sim.utils.viz_utils.semantic_to_rgb(semantic)
        semantic_img_array = np.array(pil_img)

        # Se l'immagine ha 4 canali (RGBA), rimuovi il canale alpha
        semantic_img_array = semantic_img_array[:, :, :3]

        semantic_msg = PILBridge.numpy_to_rosimg(semantic_img_array,frame_id="habitat_semantic_camera",encoding="rgb8") 
        self.semantic_pub.publish(semantic_msg)
        

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