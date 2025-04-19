#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
from PILBridge import PILBridge  # Assuming this is your custom bridge class

# Use a global flag to track callback response
received_response = False

def callback(msg):
    global received_response
    try:
        rospy.loginfo("Received segmented image")

        # Convert ROS Image to numpy grayscale image
        gray_img = PILBridge.rosimg_to_numpy(msg)

        # Normalize for visualization
        normalized = gray_img.astype(np.float32)
        normalized /= normalized.max() if normalized.max() > 0 else 1.0

        # Apply colormap (e.g. 'jet')
        colored = cm.get_cmap('jet')(normalized)[:, :, :3]  # drop alpha

        # Show with matplotlib
        plt.figure("Segmented Image (Colorized)")
        plt.imshow(colored)
        plt.axis('off')
        plt.show()

        received_response = True
        rospy.signal_shutdown("Image received and shown")

    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")

def main():
    global received_response
    rospy.init_node('segmented_image_test_node', anonymous=True)

    # Subscriber for segmented result
    rospy.Subscriber('/deeplab/segmented_image', Image, callback)

    # Publisher to send the input image
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)

    # Load the image
    img = cv2.imread('/home/michele/Desktop/Colombo/Scannet/scans/scene0002_00/color/426.jpg')
    if img is None:
        rospy.logerr("Failed to load image.")
        return

    # Convert to ROS Image
    img_msg = PILBridge.numpy_to_rosimg(img, encoding='bgr8')

    # Give some time for the subscriber to register
    rospy.sleep(1.0)
    pub.publish(img_msg)
    rospy.loginfo("Input image published to /camera/image_raw")

    # Wait for callback
    timeout = rospy.Time.now() + rospy.Duration(10)  # 10s timeout
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and not received_response and rospy.Time.now() < timeout:
        rate.sleep()

    if not received_response:
        rospy.logwarn("Timeout: No segmented image received.")
        rospy.signal_shutdown("No response received.")

if __name__ == '__main__':
    main()
