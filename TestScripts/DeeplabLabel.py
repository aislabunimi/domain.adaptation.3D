#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
from PILBridge import PILBridge
import rospkg
from LabelElaborator import LabelElaborator

# Use a global flag to track callback response
received_response = False

def callback(msg):
    global received_response
    try:
        rospy.loginfo("Received segmented image")

        rospack = rospkg.RosPack()

        this_pkg = rospack.get_path("control_node")
        mapping = np.genfromtxt(
            f"{this_pkg}/cfg/nyu40_segmentation_mapping.csv", delimiter=","
        )
            
        # Caricamento del mapping dei colori delle classi segmentate
        class_colors = mapping[1:, 1:4]
        label_elaborator = LabelElaborator(class_colors, confidence=0)


        # Convert ROS image to NumPy using PILBridge
        sem_cv = PILBridge.rosimg_to_numpy(msg)

        # Use LabelElaborator to process the NumPy image, not the ROS message
        _, colored_sem, _ = label_elaborator.process(sem_cv)
        colored_sem = cv2.cvtColor(colored_sem, cv2.COLOR_RGB2BGR)

        # Resize image to smaller size (e.g. 50%)
        scale = 0.5
        resized = cv2.resize(colored_sem, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Create a resizable window
        cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented Image", resized)

        print("Chiudi la finestra per continuare...")

        # Wait until the window is closed manually
        while cv2.getWindowProperty("Segmented Image", cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) == 27:  # Optional: ESC to exit
                break

        cv2.destroyAllWindows()

    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")
        
def main():
    global received_response
    rospy.init_node('segmented_image_test_node', anonymous=True)


    # Subscriber for segmented result
    rospy.Subscriber('/deeplab/segmented_image', Image, callback)

    # Publisher to send the input image
    pub = rospy.Publisher('/deeplab/rgb', Image, queue_size=1)

    # Load the image
    img = cv2.imread('/home/michele/Desktop/Domain-Adaptation-Pipeline/IO_pipeline/Scannet/scans/scene0002_00/color/1.jpg')
    if img is None:
        rospy.logerr("Failed to load image.")
        return

    # Convert to ROS Image
    img_msg = PILBridge.numpy_to_rosimg(img, encoding='bgr8')

    rospy.sleep(1.0)
    pub.publish(img_msg)
    rospy.loginfo("Input image published to /deeplab/rgb")

    # Wait for callback
    timeout = rospy.Time.now() + rospy.Duration(100)  # 10s timeout
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and not received_response and rospy.Time.now() < timeout:
        rate.sleep()

    if not received_response:
        rospy.logwarn("Timeout: No segmented image received.")
        rospy.signal_shutdown("No response received.")

if __name__ == '__main__':
    main()
