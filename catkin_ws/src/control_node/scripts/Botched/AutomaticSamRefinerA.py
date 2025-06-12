import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

class FastSamRefinerAuto:
    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        visualize: bool = True
    ):
        """
        Args:
            model_path: Path to the SAM model checkpoint.
            visualize: Whether to show debug plots.
        """
        self.model = SAM(model_path)
        self.visualize = visualize

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        refined_mask = np.zeros_like(mask, dtype=np.uint8)

        # Step 1: Get automatic SAM masks
        try:
            results = self.model.predict(image, mode='auto', verbose=False)
        except Exception as e:
            print(f"[ERROR] SAM auto-segmentation failed: {e}")
            return mask.copy()

        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            print("[WARN] No masks found by SAM.")
            return mask.copy()

        sam_masks = results[0].masks.data.cpu().numpy().astype(np.uint8)

        for sam_mask in sam_masks:
            overlapping_labels = mask[sam_mask == 1]
            valid_labels = overlapping_labels[overlapping_labels != 0]

            if valid_labels.size == 0:
                continue

            majority_label = np.bincount(valid_labels).argmax()
            refined_mask[sam_mask == 1] = majority_label

            if self.visualize:
                self._debug_mask(image, sam_mask, majority_label)

        # Step 2: Fill unassigned space using "ereditary" strategy
        unassigned = (refined_mask == 0) & (mask != 0)
        refined_mask[unassigned] = mask[unassigned]

        # Step 3: Optional visualization
        if self.visualize:
            self._debug_final(image, refined_mask)

        return refined_mask

    def _debug_mask(self, image, sam_mask, label):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(sam_mask, cmap='gray')
        axes[1].set_title(f"SAM Mask - Label {label}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    def _debug_final(self, image, mask):
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.title("Final Refined Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
