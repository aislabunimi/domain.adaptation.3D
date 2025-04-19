#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as SensorImage
import habitat_sim
import numpy as np
from std_msgs.msg import Float64
from Modules import PILBridge
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from habitat_ros_bridge.msg import Sensors

test_scene = "/home/michele/Desktop/Colombo/HM3D/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
semantic_test_scene = "/home/michele/Desktop/Colombo/HM3D/minival/00800-TEEsavR23oF/TEEsavR23oF.semantic.glb"
scene_config = "/home/michele/Desktop/Colombo/HM3D/hm3d_annotated_basis.scene_dataset_config.json"

train_scene = "/home/michele/Desktop/Colombo/HM3D/train/00000-kfPV7w3FaU5/kfPV7w3FaU5.glb"

settings = {
    "scene": test_scene,  # Scene path
    "semantic": semantic_test_scene,
    "scene_conf": scene_config,
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.7,  # Height of sensors in meters, relative to the agent
    "width": 640,         # Image width
    "height": 480,        # Image height
}

def setup_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_conf"]
   # sim_cfg.load_semantic_mesh = True

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
        self.scene_pub = rospy.Publisher("/habitat/scene/sensors", sensors, queue_size=10)
        self.semantic_pub = rospy.Publisher("/habitat/semantic", SensorImage, queue_size=10)

    
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

    def print_scene_recur(scene, limit_output=10):
        print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        count = 0
        for level in scene.levels:
            print(
                f"Level id:{level.id}, center:{level.aabb.center},"
                f" dims:{level.aabb.sizes}"
            )
            for region in level.regions:
                print(
                    f"Region id:{region.id}, category:{region.category.name()},"
                    f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                )
                for obj in region.objects:
                    print(
                        f"Object id:{obj.id}, category:{obj.category.name()},"
                        f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                    )
                    count += 1
                    if count >= limit_output:
                        return None

    def publish_images(self):
        obs = self.sim.get_sensor_observations()
        
        # Publish RGB image
        rgb = obs["color_sensor"][:, :, :3]  # Extract RGB data
        rgb_msg =pilBridge.numpy_to_rosimg(
            rgb,
            frame_id="habitat_rgb_camera",
            encoding="rgb8"
        )
        self.rgb_pub.publish(rgb_msg)
        
        # Publish Depth image
        depth = obs["depth_sensor"]
        depth_32fc1 = depth.astype(np.float32)
        depth_msg = pilBridge.numpy_to_rosimg(
            depth_32fc1,
            frame_id="habitat_depth_camera",
            encoding="32FC1"
        )
        self.depth_pub.publish(depth_msg)

                
        # Publish semantic image
        semantic = obs["semantic_sensor"]
        pil_img = habitat_sim.utils.viz_utils.semantic_to_rgb(semantic)
     
        semantic_img_array = np.array(pil_img)
        # Se l'immagine ha 4 canali (RGBA), rimuovi il canale alpha
        semantic_img_array = semantic_img_array[:, :, :3]

        semantic_msg = pilBridge.numpy_to_rosimg(
            semantic_img_array,
            frame_id="habitat_semantic_camera",
            encoding="rgb8"
        ) 

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