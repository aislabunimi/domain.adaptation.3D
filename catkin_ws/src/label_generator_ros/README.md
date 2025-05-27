# ROS Ray Cast Scene Node (`label_generator_node`)

This ROS node performs **ray casting from a 3D mesh** to generate **2D semantic label images** based on a given camera pose.

It is adapted from the KimeraInterfacer/pseudo_labels project and customized for seamless integration within our ROS-based pipeline.

The node utilizes a serialized 3D map and camera intrinsics to simulate semantic labeling from a virtual viewpoint in a known environment.

---

## Initialization (One-Time Setup)

Before generating labels, the node must be initialized with camera parameters and scene data.

### Topic: `/label_generator/init`  
**Type:** `label_generator_ros/LabelGenInit`

This topic initializes the ray casting engine. It must be published once before any labeling requests are sent.

#### Message Fields

| Field                    | Type        | Description                                                  |
|--------------------------|-------------|--------------------------------------------------------------|
| `height`                 | `uint32`    | Height of the output label image (in pixels)                |
| `width`                  | `uint32`    | Width of the output label image (in pixels)                 |
| `k_image`                | `float64[9]`| Row-major 3×3 intrinsic camera matrix                       |
| `mesh_path`              | `string`    | File path to the 3D mesh of the environment                 |
| `map_serialized_path`    | `string`    | File path to the serialized semantic map                    |

> **Note:** The `k_image` array should be the flattened version of the 3×3 camera intrinsic matrix:  
> ```
> [ fx,  0,  cx,
>    0, fy,  cy,
>    0,  0,   1 ]
> ```

---

## Label Request

After initialization, label images can be requested by sending a camera pose.

### Topic: `/label_generator/request`  
**Type:** `std_msgs/Float64MultiArray`

#### Message Content
- A **flattened 4×4 transformation matrix** representing the camera pose in the world frame.
- This should include both position and orientation of the camera.

---

## Output: Labeled Image

The node publishes the resulting label image to:

### Topic: `/label_generator/label`  
**Type:** `label_generator_ros/LabelGenResponse`

#### Message Fields

| Field       | Type                  | Description                                      |
|-------------|-----------------------|--------------------------------------------------|
| `label`     | `sensor_msgs/Image`   | Output mono8 image, with pixel-wise class labels |
| `success`   | `bool`                | True if labeling succeeded                       |
| `error_msg` | `string`              | Empty on success, contains error info if failed  |

The `label` image is a grayscale image where each pixel's value corresponds to a semantic class ID.