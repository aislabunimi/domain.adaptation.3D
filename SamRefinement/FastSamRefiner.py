import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

class SAM2RefinerMixed:
    def __init__(self, model_path: str = "sam2_b.pt", visualize: bool = True, skip_labels: list = None):
        """
        Initializes the SAM2 model.
        
        Args:
            model_path (str): Path to SAM2 model weights.
            visualize (bool): Whether to show debug visualization.
            skip_labels (list): List of class indices to skip during refinement.
        """
        self.model = SAM(model_path)
        self.visualize = visualize
        self.skip_labels = skip_labels if skip_labels else []

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refines a segmentation mask using SAM2 with batched points and bounding boxes as prompts.
        Skips background (label=0) and any labels in `self.skip_labels`.
        """
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)
        print(f"[INFO] Found labels: {unique_labels}")

        all_points = []
        all_bboxes = []
        prompt_labels = []

        # Step 1: Collect prompts
        for label in unique_labels:
            if label == 0 or label in self.skip_labels:
                continue  # Skip background and ignored classes

            binary_mask = (mask == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            num_components, components = cv2.connectedComponents(dilated_mask)

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)
                ys, xs = np.where(comp_mask == 1)
                if len(xs) == 0:
                    continue

                cx, cy = int(np.median(xs)), int(np.median(ys))
                x, y, w, h = cv2.boundingRect(comp_mask)
                x1, y1, x2, y2 = x, y, x + w, y + h

                all_points.append([cx, cy])
                all_bboxes.append([x1, y1, x2, y2])
                prompt_labels.append(label)

        if not all_points:
            print("[WARN] No valid prompts found.")
            return mask.copy()

        # Step 2: Batch predict
        try:
            results = self.model.predict(
                image,
                points=all_points,
                bboxes=all_bboxes,
                verbose=False
            )
        except Exception as e:
            print(f"[ERROR] Batch SAM prediction failed: {e}")
            return mask.copy()

        # Step 3: Process results
        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            print(f"[WARN] No masks returned from SAM.")
            return mask.copy()

        for idx, sam_m in enumerate(results[0].masks):
            sam_mask = sam_m.data.cpu().numpy().squeeze().astype(np.uint8)
            overlapping_labels = mask[sam_mask == 1]
            if overlapping_labels.size == 0:
                continue

            majority_label = np.bincount(overlapping_labels).argmax()
            refined_mask[sam_mask == 1] = majority_label

        # Fill unassigned areas
        refined_mask[refined_mask == 0] = mask[refined_mask == 0]
        print("[INFO] Refinement complete.")

        # Optional visualization
        if self.visualize:
            self._debug_visualize(image, refined_mask, all_points, all_bboxes)

        return refined_mask

    def _debug_visualize(self, image, mask, points, bboxes):
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        for (x, y) in points:
            plt.plot(x, y, 'ro', markersize=3)
        for (x1, y1, x2, y2) in bboxes:
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', linewidth=1))
        plt.imshow(mask, alpha=0.4, cmap='jet')
        plt.title("Refinement Debug View")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
