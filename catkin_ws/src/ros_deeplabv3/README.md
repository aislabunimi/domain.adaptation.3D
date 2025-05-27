# ROS DeepLabV3 Package

This package provides two main ROS nodes for real-time semantic segmentation and model fine-tuning using DeepLabV3:

- `deeplab_segmenter`: Performs real-time semantic segmentation on incoming images.
- `deeplab_finetune_service`: Exposes a ROS service to fine-tune the DeepLabV3 model.

---

## Node: `deeplab_segmenter`

### Description
Performs semantic segmentation in real-time using a pretrained or fine-tuned DeepLabV3 model.

### Input
- **Topic:** `/camera/image_raw`  
- **Type:** `sensor_msgs/Image`  
- **Description:** Input RGB image stream.

### Output
- **Topic:** `/deeplab/segmented_image`  
- **Type:** `sensor_msgs/Image` (`mono8`)  
- **Description:** Output segmentation map, with each pixel labeled according to the predicted semantic class.

---

## Service Node: `deeplab_finetune_service`

### Description
Provides a ROS service to fine-tune the DeepLabV3 model on a new dataset of images and segmentation masks.

### Service Details
- **Service Name:** `/deeplab_finetune`  
- **Service Type:** `Finetune.srv`

#### `Finetune.srv` Definition
```srv
string dataset_path        # Path to directory containing training images and masks
uint32 num_epochs          # Number of training epochs
uint32 num_classes         # Number of semantic classes
---
bool success               # Indicates whether fine-tuning completed successfully
string message             # Informational or error message
