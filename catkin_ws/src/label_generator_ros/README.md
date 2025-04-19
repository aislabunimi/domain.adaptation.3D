# Ray Cast Scene Node (`label_generator_node`)

This ROS node performs **ray casting from a 3D mesh** to generate **2D label images** given a camera pose.

This node is based on the KimeraInterfacer/pseudo_labels and has been customized for use with our pipeline and ROS integration.

It uses a serialized map and camera intrinsics to simulate image labeling in a known 3D environment.

---

## Initialization (one-time)

Before sending labeling requests, the node must be initialized via:

### Topic: `/label_generator/init`
**Type:** `label_generator_ros/LabelGenInit`

This message initializes the ray casting engine and must be sent once before any labeling request.

#### **Message structure:**

| Field                | Type       | Description                                                  |
|----------------------|------------|--------------------------------------------------------------|
| `uint32 height`      | uint32     | Height of the output image in pixels                         |
| `uint32 width`       | uint32     | Width of the output image in pixels                          |
| `float64[9] k_image` | float64[]  | 3×3 row-major camera intrinsic matrix (K)                    |
| `string mesh_path`   | string     | Path to the 3D mesh file of the scene                        |
| `string map_serialized_path` | string | Path to serialized semantic map |

> **Note:** `k_image` should be the flattened version of the intrinsic matrix:
> ```
> fx  0  cx
> 0  fy  cy
> 0   0   1
> ```

---

## Requesting Labels

Once initialized, you can request label generation via:

### Topic: `/label_generator/request`  
**Type:** `std_msgs/Float64MultiArray`

#### **Message structure:**
- A flattened 4×4 pose matrix (`Float64MultiArray.data`)
- The pose should represent the camera's position and orientation in the world frame.

---

## Published Labels

The node responds by publishing the label image to:

### Topic: `/label_generator/label`  
**Type:** `label_generator_ros/LabelGenResponse`

#### **Message structure:**

| Field            | Type              | Description                                                  |
|------------------|-------------------|--------------------------------------------------------------|
| `label`          | `sensor_msgs/Image` | The ray-casted semantic label image (if successful)          |
| `success`        | `bool`            | True if label was generated successfully                     |
| `error_msg`      | `string`          | Empty if success, otherwise a description of the error       |

The `label` image is of type `mono8`, where pixel values represent semantic class IDs.