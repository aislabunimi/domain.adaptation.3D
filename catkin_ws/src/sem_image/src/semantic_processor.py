#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as SensorImage
from TestScripts.PILBridge import PILBridge
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image as PILImage
import habitat_sim


class SemanticProcessor:
    def __init__(self):
        rospy.init_node('semantic_processor', anonymous=True)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        
        rospy.Subscriber("/habitat/rgb", SensorImage, self.rgb_callback)
        self.rgb_pub = rospy.Publisher("/habitat/semantic_prediction", SensorImage, queue_size=10)


        rospy.spin()

    def rgb_callback(self, msg):

        input_image = PILBridge.rosimg_to_numpy(msg)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
            #rospy.loginfo(output.shape)

        output_predictions = output.argmax(0)
        #rospy.loginfo(output_predictions)

        # Converti le previsioni in uint8
        output_predictions = output_predictions.cpu().numpy().astype("uint8")


        pil_img = habitat_sim.utils.viz_utils.semantic_to_rgb(output_predictions)

        semantic_img_array = np.array(pil_img)
        # Se l'immagine ha 4 canali (RGBA), rimuovi il canale alpha
        semantic_img_array = semantic_img_array[:, :, :3]

        semantic_msg = PILBridge.numpy_to_rosimg(
            semantic_img_array,
            frame_id="habitat_semantic_processor_camera",
            encoding="rgb8"
        )

        self.rgb_pub.publish(semantic_msg)


if __name__ == "__main__":
    try:
        SemanticProcessor()
    except rospy.ROSInterruptException:
        pass