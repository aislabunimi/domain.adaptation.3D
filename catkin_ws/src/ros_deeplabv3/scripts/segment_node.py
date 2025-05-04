#!/usr/bin/env python3

import rospy
import numpy as np
import os
import torchvision
import torch
import torchvision.transforms as T
from torchvision import models

from PIL import Image as PILImage
from Modules import PILBridge
from sensor_msgs.msg import Image
import cv2


class DeepLabSegmenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        #self.model = models.segmentation.deeplabv3_resnet101(num_classes=41, aux_loss=True)

        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=False,
            pretrained_backbone=True,
            progress=True,
            num_classes=40,
            aux_loss=None,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        finetuned_path = os.path.join(base_dir, "..", "models", "deeplabv3.pth")

        if os.path.exists(finetuned_path):
            rospy.loginfo("Loading finetuned model...")
            self.model.load_state_dict(torch.load(finetuned_path, map_location=self.device))
            rospy.loginfo("Model loaded")
        else:
            rospy.logwarn("Finetuned model not found. Using untrained model! in "+ finetuned_path)

        self.model.to(self.device)
        self.model.eval()


        self.base_transform = T.Compose([
            T.Resize((240, 320), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])

        rospy.Subscriber('/deeplab/rgb', Image, self.callback)
        self.pub = rospy.Publisher("/deeplab/segmented_image", Image, queue_size=1)

    def callback(self, msg):
        try:
            np_img = PILBridge.PILBridge.rosimg_to_numpy(msg)

            original_h, original_w = np_img.shape[:2]

            if np_img.ndim == 2:  # grayscale
                np_img = np.stack([np_img] * 3, axis=-1)
            elif np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]  # drop alpha if present

            pil_img = PILImage.fromarray(np_img)

            input_tensor = self.base_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                pred = torch.argmax(output, dim=0).byte().cpu().numpy()
                pred += 1
            torch.cuda.empty_cache()
            # Resize prediction back to original image size
            pred_resized = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            ros_seg = PILBridge.PILBridge.numpy_to_rosimg(
                pred_resized.astype(np.uint8),
                encoding="mono8",
                frame_id=msg.header.frame_id,
                stamp=msg.header.stamp
            )
            self.pub.publish(ros_seg)

        except Exception as e:
            rospy.logerr(f"Segmentation error: {e}")

if __name__ == '__main__':
    rospy.init_node('deeplab_segmenter')
    DeepLabSegmenter()
    rospy.spin()
