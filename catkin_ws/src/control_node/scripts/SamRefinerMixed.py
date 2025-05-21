import cv2
import numpy as np
from ultralytics import SAM

class SAM2RefinerMixed:
    def __init__(self, model_path: str = "sam2_b.pt"):
        """
        Initializes the SAM2 model.
        """
        self.model = SAM(model_path)

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refines a segmentation mask using SAM2 with both points and bounding boxes as prompts.
        Skips wall=1 and floor=2.
        """
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)
        print(f"[INFO] Found labels: {unique_labels}")

        for label in unique_labels:
            if label in [0]:
                continue  # Skip background, wall, floor

            binary_mask = (mask == label).astype(np.uint8)

            # Connect nearby fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Connected components
            num_components, components = cv2.connectedComponents(dilated_mask)
            #print(f"[INFO] Label {label} → {num_components - 1} connected components")

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)

                # Bounding box
                x, y, w, h = cv2.boundingRect(comp_mask)
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Median point inside the mask
                ys, xs = np.where(comp_mask == 1)
                if len(xs) == 0:
                    continue
                cx, cy = int(np.median(xs)), int(np.median(ys))

                try:
                    results = self.model.predict(
                        image,
                        points=[[cx, cy]],
                        bboxes=[ [x1, y1, x2, y2]],
                        verbose = False            
                    )
                except Exception as e:
                    print(f"[ERROR] SAM prediction failed for box ({x1}, {y1}, {x2}, {y2}): {e}")
                    continue

                if not results or not hasattr(results[0], "masks") or results[0].masks is None:
                    print(f"[WARN] No masks returned for label {label} at ({cx}, {cy})")
                    continue

                for sam_m in results[0].masks:
                    sam_mask = sam_m.data.cpu().numpy().squeeze().astype(np.uint8)

                    # Determine majority label
                    overlapping_labels = mask[sam_mask == 1]
                    if overlapping_labels.size == 0:
                        continue
                    majority_label = np.bincount(overlapping_labels).argmax()

                    refined_mask[sam_mask == 1] = majority_label

        # Fill holes with original mask
        refined_mask[refined_mask == 0] = mask[refined_mask == 0]
        print("[INFO] Refinement complete.")
        return refined_mask
