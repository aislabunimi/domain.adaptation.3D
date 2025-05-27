# Habitat ROS Bridge

This ROS node creates a simple bridge between [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and ROS.

It sets up a simulation with an agent equipped with:
- An RGB camera
- A depth camera
- A semantic camera

---

## Subscribed Topics

- **`/cmd_vel`** (`geometry_msgs/Twist`):  
  Converts velocity commands into agent actions. The following mappings are applied:

  | Twist input            | Habitat action     |
  |------------------------|--------------------|
  | `linear.x > 0`         | `move_forward`     |
  | `linear.x < 0`         | `move_backward`    |
  | `linear.y > 0`         | `move_up`          |
  | `linear.y < 0`         | `move_down`        |
  | `angular.z > 0`        | `turn_left`        |
  | `angular.z < 0`        | `turn_right`       |
  | No movement            | `stop`             |

---

## Published Topics

- **`/habitat/scene/sensors`** (`habitat_ros_bridge/Sensors`):  
  Custom message containing:
  - `sensor_msgs/Image rgb`: RGB image from the agent's color sensor
  - `sensor_msgs/Image depth`: Depth image from the agent's depth sensor

- **`/habitat/semantic`** (`sensor_msgs/Image`):  
  RGB image representing the semantic view from the agent's semantic sensor.


---

## 🛠 Custom Message: `Sensors.msg`

```msg
sensor_msgs/Image rgb
sensor_msgs/Image depth
