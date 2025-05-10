# Kimera Interfacer Suite for ScanNet Dataset

> **Note:** Before using these nodes, make sure to check the corresponding launch files and any hard-coded static file paths in the scripts.

---

## 1. DLMOCK - ScanNet Dataset Publisher

The `dlmock` node is responsible for loading data from the **ScanNet** dataset and publishing it as ROS messages. It serves as the dataset interface that emulates real-time sensor data streaming into ROS.

### Published Topics

| Topic Name            | Message Type             | Description                                    |
|-----------------------|--------------------------|------------------------------------------------|
| `/depth_topic`        | `sensor_msgs/Image`      | Publishes depth images from the dataset.       |
| `/image_topic`        | `sensor_msgs/Image`      | Publishes RGB images.                          |
| `/seg_topic`          | `sensor_msgs/Image`      | Publishes semantic segmentation images.        |
| `/sync_topic`         | `kimera_interfacer/SyncSemantic` | Publishes synchronized RGB, depth, and semantic data. |

#### SyncSemantic Message Structure

| Field     | Type                | Description                     |
|-----------|---------------------|---------------------------------|
| `image`   | `sensor_msgs/Image` | RGB image                       |
| `depth`   | `sensor_msgs/Image` | Depth image                     |
| `sem`     | `sensor_msgs/Image` | Semantic segmentation image     |

### Camera Info Topics

| Topic Name              | Message Type             | Description                                      |
|-------------------------|--------------------------|--------------------------------------------------|
| `/rgb_camera_info`      | `sensor_msgs/CameraInfo` | Camera intrinsics for the RGB camera.            |
| `/depth_camera_info`    | `sensor_msgs/CameraInfo` | Camera intrinsics for the depth camera.          |

---

## 2. Kimera Interfacer - Mesh Generation Node

The `kimera_interfacer` node processes data from the `dlmock` node and generates a **semantic 3D mesh**. It uses the `SemanticTsdfServer` to fuse depth and semantic information into a volumetric TSDF map and outputs both TSDF/ESDF maps and an optional `.ply` mesh.

### Subscribed Topics

| Topic Name                                | Message Type                   | Description                                           |
|-------------------------------------------|--------------------------------|-------------------------------------------------------|
| `/kimera_interfacer/sync_semantic`        | `kimera_interfacer/SyncSemantic` | Receives synchronized RGB, depth, and semantic images. |
| `/kimera_interfacer/end_generation_map`   | `std_msgs/Bool`                | Stops mesh generation when message is `true`.         |
| `/depth_camera_info`                      | `sensor_msgs/CameraInfo`       | Retrieves depth camera intrinsics.                    |

### Published Topics

| Topic Name                 | Message Type       | Description                             |
|----------------------------|--------------------|-----------------------------------------|
| `/kimera_interfacer/pcl`   | `kimera::PointCloud` | Publishes the generated semantic point cloud. |

---

## Processing Workflow

1. **Data Synchronization**  
   Receives RGB, depth, and semantic images from the `SyncSemantic` message. These are used to generate a point cloud from depth and semantics.

2. **Point Cloud Generation**  
   Depth and semantic images are converted to a semantic point cloud using intrinsic parameters and published.

3. **TF Transformation**  
   Uses `tf2_ros` to transform the point cloud to the base frame for insertion into the TSDF map.

4. **Semantic TSDF Integration**  
   The point cloud is integrated into a TSDF using Kimera's `SemanticTsdfServer`.

5. **Map Generation and Saving**  
   Upon receiving a stop signal:
   - TSDF and semantic layers are serialized.
   - A mesh is generated and saved in `.ply` format if `mesh_filename` is set.
   - The ESDF layer is computed in batch and saved alongside the TSDF.