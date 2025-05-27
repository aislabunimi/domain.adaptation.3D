#!/usr/bin/env python3
import rospy
import random
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as SensorImage
from Modules.PILBridge import PILBridge

class RandomNavigator:
    def __init__(self):
        rospy.init_node('random_navigator', anonymous=True)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/habitat/rgb", SensorImage, self.rgb_callback)
        rospy.Subscriber("/habitat/depth", SensorImage, self.depth_callback)

        # Navigation parameters
        self.obstacle_distance = 0.8  # meters
        self.min_side_distance = 0.5  # meters
        self.turn_duration = 2.0     # seconds
        self.forward_speed = 0.1
        self.turn_speed = 0.5
        
        # State variables
        self.current_depth = None
        self.last_turn_time = rospy.Time.now()
        self.current_rgb = None
        self.rate = rospy.Rate(1/20)
        
        self.navigate()

    def rgb_callback(self, msg):
        # Convert ROS Image to NumPy array using PILBridge
        self.current_rgb = PILBridge.rosimg_to_numpy(msg)
        # Convert BGR to RGB if needed (depending on your image source)
        # self.current_rgb = self.current_rgb[:, :, ::-1]  # Uncomment if BGR->RGB needed

    def depth_callback(self, msg):
        # Convert ROS Image to NumPy array using PILBridge
        self.current_depth = PILBridge.rosimg_to_numpy(msg)
        # Ensure depth is float32 (might be needed depending on your setup)
        self.current_depth = self.current_depth.astype(np.float32)

    def get_obstacle_info(self):
        if self.current_depth is None:
            return False, (False, False)

        height, width = self.current_depth.shape

        # **New depth analysis - Focus more on the lower part**
        lower_start = int(height * 0.6)  # Only look at the bottom 40%
        ground_start = int(height * 0.85)  # Lowest 15% for ground obstacles

        center_region = self.current_depth[lower_start:, int(width * 0.3):int(width * 0.7)]
        left_region = self.current_depth[lower_start:, int(width * 0.1):int(width * 0.3)]
        right_region = self.current_depth[lower_start:, int(width * 0.7):int(width * 0.9)]
        
        # **Additional Ground-Level Check**
        ground_region = self.current_depth[ground_start:, :]

        # **Use median instead of min to filter noise**
        min_center = np.median(center_region)
        min_left = np.median(left_region)
        min_right = np.median(right_region)
        ground_obstacle = np.median(ground_region) < self.obstacle_distance

        # **Detect obstacles**
        obstacle_detected = (min_center < self.obstacle_distance) or ground_obstacle
        left_clear = min_left > self.min_side_distance
        right_clear = min_right > self.min_side_distance

        return obstacle_detected, (left_clear, right_clear)

    def navigate(self):

        rotating = False
        rotation_direction = 0  # +1 for left, -1 for right
        total_rotation = 0
        last_angle = 0  # Track previous angle for rotation calculations

        while not rospy.is_shutdown():
            twist = Twist()
            obstacle_detected, (left_clear, right_clear) = self.get_obstacle_info()

            if obstacle_detected:
                if not rotating:
                    # Start rotating: decide direction based on available space
                    if left_clear and right_clear:
                        rotation_direction = random.choice([-1, 1])  # Random left or right
                    elif left_clear:
                        rotation_direction = 1  # Turn left
                    elif right_clear:
                        rotation_direction = -1  # Turn right
                    else:
                        rotation_direction = random.choice([-1, 1])  # No clear path, random turn

                    rotating = True
                    total_rotation = 0  # Reset rotation counter
                    last_angle = rospy.Time.now().to_sec()

                # Rotate in the chosen direction
                twist.angular.z = rotation_direction * self.turn_speed
                twist.linear.x = 0.0

                # Check rotation progress
                current_time = rospy.Time.now().to_sec()
                time_elapsed = current_time - last_angle
                total_rotation += abs(self.turn_speed) * time_elapsed
                last_angle = current_time

                # If a full rotation (360° or 2π radians) is completed, mark as stuck
                if total_rotation >= 2 * np.pi:
                    rospy.logwarn("STUCK")
                    rotating = False  # Stop rotating (you might want to add an alternative behavior here)

            else:
                rotating = False  # Stop rotation if path is clear
                twist.linear.x = self.forward_speed
                twist.angular.z = 0.0  # Ensure it stops turning

            self.cmd_pub.publish(twist)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        RandomNavigator()
    except rospy.ROSInterruptException:
        pass