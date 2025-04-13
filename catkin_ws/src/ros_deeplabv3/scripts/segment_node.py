#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image as SensorImage
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import os
from PIL import Image as PILImage
from PILBridge import PILBridge  # make sure it's in the same folder or properly installed

class DeepLabSegmenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)  # Adjust for num_classes

        finetuned_path = os.path.expanduser("~/deeplab_finetuned.pth")
        if os.path.exists(finetuned_path):
            rospy.loginfo("Loading finetuned model...")
            self.model.load_state_dict(torch.load(finetuned_path, map_location=self.device))
        else:
            rospy.logwarn("Finetuned model not found. Using untrained model!")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((520, 520)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        rospy.Subscriber("/camera/image_raw", SensorImage, self.callback)
        self.pub = rospy.Publisher("/segmented_image", SensorImage, queue_size=1)

    def callback(self, msg):
        try:
            # Convert ROS Image to NumPy using PILBridge
            np_img = PILBridge.rosimg_to_numpy(msg)

            if np_img.ndim == 2:  # grayscale
                np_img = np.stack([np_img] * 3, axis=-1)
            elif np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]  # drop alpha if present

            pil_img = PILImage.fromarray(np_img)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                pred = torch.argmax(output, dim=0).byte().cpu().numpy()

            # Convert predicted mask to color image (simple colormap)
            ros_seg = PILBridge.numpy_to_rosimg(pred.astype(np.uint8), encoding="mono8", frame_id=msg.header.frame_id, stamp=msg.header.stamp)

            self.pub.publish(ros_seg)

        except Exception as e:
            rospy.logerr(f"Segmentation error: {e}")

if __name__ == '__main__':
    rospy.init_node('deeplab_segmenter')
    DeepLabSegmenter()
    rospy.spin()
