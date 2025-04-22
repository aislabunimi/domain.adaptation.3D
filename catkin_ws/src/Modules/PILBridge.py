#!/usr/bin/env python3
import rospy
import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image as SensorImage

class PILBridge:
    @staticmethod
    def numpy_to_rosimg(array, frame_id="camera", encoding="rgb8", stamp=None):
        """
        Convert a NumPy array to a ROS Image message using PIL
        Args:
            array: NumPy array (H,W,3), (H,W), or (H,W) uint16 for 16UC1
            frame_id: ROS frame ID for the image header
            encoding: ROS image encoding (e.g., "rgb8", "bgr8", "mono8", "32FC1", "16UC1")
            stamp: ROS Time for the header (default: current time)
        Returns:
            sensor_msgs.msg.Image
        """
        if stamp is None:
            stamp = rospy.Time.now()

        img_msg = SensorImage()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame_id
        img_msg.height, img_msg.width = array.shape[:2]

        if encoding is None:
            if array.ndim == 2:
                encoding = "mono8" if array.dtype == np.uint8 else "16UC1" if array.dtype == np.uint16 else "32FC1"
            elif array.shape[2] == 3:
                encoding = "rgb8"
            else:
                raise ValueError("Cannot infer encoding from array shape/dtype")

        img_msg.encoding = encoding

        # Set step size and convert to bytes accordingly
        if encoding in ["rgb8", "bgr8"]:
            img_msg.step = img_msg.width * 3
            img_msg.data = array.tobytes()
        elif encoding == "mono8":
            img_msg.step = img_msg.width
            img_msg.data = array.tobytes()
        elif encoding == "32FC1":
            img_msg.step = img_msg.width * 4
            img_msg.data = array.astype(np.float32).tobytes()
        elif encoding == "16UC1":
            img_msg.step = img_msg.width * 2
            img_msg.data = array.astype(np.uint16).tobytes()
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        img_msg.is_bigendian = 0
        return img_msg

    @staticmethod
    def rosimg_to_numpy(img_msg):
        """
        Convert a ROS Image message to a NumPy array using PIL or raw bytes
        Args:
            img_msg: sensor_msgs.msg.Image
        Returns:
            NumPy array
        """
        if img_msg.encoding == "16UC1":
            return np.frombuffer(img_msg.data, dtype=np.uint16).reshape((img_msg.height, img_msg.width))
        elif img_msg.encoding == "32FC1":
            return np.frombuffer(img_msg.data, dtype=np.float32).reshape((img_msg.height, img_msg.width))
        else:
            # Use PIL for supported encodings
            pil_img = PILImage.frombytes(
                mode=PILBridge._ros_encoding_to_pil_mode(img_msg.encoding),
                size=(img_msg.width, img_msg.height),
                data=img_msg.data,
                decoder_name='raw'
            )
            return np.array(pil_img)

    @staticmethod
    def _ros_encoding_to_pil_mode(encoding):
        """
        Map ROS image encoding to PIL mode
        """
        encoding_map = {
            "rgb8": "RGB",
            "bgr8": "RGB",  # Channel order needs handling
            "mono8": "L",
            "32FC1": "F",
            "rgba8": "RGBA",
            "bgra8": "RGBA"  # Channel order needs handling
            # "16UC1" is not supported by PIL, handled separately
        }

        if encoding not in encoding_map:
            raise ValueError(f"Unsupported ROS encoding: {encoding}")
        
        return encoding_map[encoding]
