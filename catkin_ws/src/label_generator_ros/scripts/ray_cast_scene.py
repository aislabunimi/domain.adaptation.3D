#!/usr/bin/env python3

import rospy
import numpy as np
from Modules import PILBridge
from sensor_msgs.msg import Image
from label_generator_ros.srv import InitLabelGenerator, InitLabelGeneratorResponse
from label_generator_ros.srv import GenerateLabel, GenerateLabelResponse
from label_gen.label_generation_ros import LabelGenerator

class LabelGenNode:
    def __init__(self):
        rospy.init_node('label_generator_node')

        self.label_generator = None

        rospy.Service('/label_generator/init', InitLabelGenerator, self.handle_init)
        rospy.Service('/label_generator/generate', GenerateLabel, self.handle_generate)

        rospy.loginfo("Label Generator Service ready.")
        rospy.spin()

    def handle_init(self, req):
        try:
            k_image = np.array(req.k_image).reshape(3, 3)
            self.label_generator = LabelGenerator(
                image_shape=(req.height, req.width),
                k_color=k_image,
                mesh_path=req.mesh_path,
                map_serialized_path=req.map_serialized_path,
                r_sub=1,
                visu3d=False
            )
            rospy.loginfo("Label Generator initialized.")
            return InitLabelGeneratorResponse(success=True, error_msg="")
        except Exception as e:
            rospy.logerr(f"Initialization failed: {e}")
            return InitLabelGeneratorResponse(success=False, error_msg=str(e))

    def handle_generate(self, req):
        if self.label_generator is None:
            return GenerateLabelResponse(
                label=Image(),
                success=False,
                error_msg="Label generator not initialized."
            )
        try:
            pose = np.array(req.pose).reshape(4, 4)
            probs = self.label_generator.get_label(pose)
            label = np.argmax(probs[:, :, 1:], axis=-1).astype(np.uint8)

            label_msg = PILBridge.PILBridge.numpy_to_rosimg(label, encoding="mono8")
            label_msg.header.stamp = rospy.Time.now()  # Handle timestamp here

            return GenerateLabelResponse(
                label=label_msg,
                success=True,
                error_msg=""
            )
        except Exception as e:
            rospy.logerr(f"Label generation failed: {e}")
            return GenerateLabelResponse(
                label=Image(),
                success=False,
                error_msg=str(e)
            )

if __name__ == "__main__":
    try:
        LabelGenNode()
    except rospy.ROSInterruptException:
        pass
