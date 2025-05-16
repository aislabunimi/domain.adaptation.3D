import numpy as np
import imageio
import torch

__all__ = ['LabelElaborator']

class LabelElaborator:
    def __init__(self, color_map, confidence=0):
        self._confidence = confidence
        self.max_classes = 40  # Including class 0
        self.rgb = color_map

        # Precompute mask for probabilistic bit extraction
        iu16 = np.iinfo(np.uint16)
        mask = np.full((1, 1), iu16.max, dtype=np.uint16)
        self.mask_low = np.right_shift(mask, 6, dtype=np.uint16)[0, 0]

    def process(self, img):
        """
        Accepts a segmented image (loaded as np.ndarray).
        Returns: (label_indices (H, W), colored_RGB_image (H, W, 3), method)
        """
        if len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA probabilistic encoding
            class_map = np.zeros((img.shape[0], img.shape[1], self.max_classes))
            for i in range(3):
                prob = np.bitwise_and(img[:, :, i], self.mask_low) / 1023
                cls = np.right_shift(img[:, :, i], 10, dtype=np.uint16)
                m = np.eye(self.max_classes)[cls] == 1
                class_map[m] = prob.reshape(-1)
            m = np.max(class_map, axis=2) < self._confidence
            label = np.argmax(class_map, axis=2).astype(np.int32)
            label[m] = 0
            method = "RGBA"

        elif len(img.shape) == 2 and img.dtype == np.uint8:
            label = img.astype(np.int32)
            method = "FAST"

        elif len(img.shape) == 2 and img.dtype == np.uint16:
            label_tensor = torch.from_numpy(img.astype(np.int32)).type(torch.float32)[None, :, :]
            shape = label_tensor.shape
            label_tensor = label_tensor.flatten()
            label_tensor = label_tensor.type(torch.int64)
            label = label_tensor.numpy().reshape(shape).astype(np.int32)[0]
            method = "MAPPED"

        else:
            raise Exception(f"Unsupported image type: shape={img.shape}, dtype={img.dtype}")

        return label, self.labels_to_rgb(label), method

    def labels_to_rgb(self, label_img):
        """Convert label indices (H, W) to RGB image using predefined colormap."""
        sem_new = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
        for i in range(self.max_classes):  # self.max_classes is 41
            sem_new[label_img == i] = self.rgb[i]
        return sem_new

