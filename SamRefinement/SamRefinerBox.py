import cv2
import numpy as np
from ultralytics import SAM
import matplotlib.pyplot as plt

class SAM2RefinerBox:
    def __init__(self, model_path: str = "sam2_b.pt"):
        """
        Initializes the SAM2 model.
        """
        self.model = SAM(model_path)

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refines a segmentation mask using SAM2 with bounding box prompts.
        Each connected component of each class (except background) is treated individually.
        """
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)
        print(f"[INFO] Found labels: {unique_labels}")

        for label in unique_labels:
            if label in [0]:  # Skip background
                continue

            binary_mask = (mask == label).astype(np.uint8)

            # Connect nearby fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # Extract connected components
            num_components, components = cv2.connectedComponents(dilated_mask)
            print(f"[INFO] Label {label} → {num_components - 1} connected components")

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)

                # Get bounding box of the component
                x, y, w, h = cv2.boundingRect(comp_mask)
                x1, y1, x2, y2 = x, y, x + w, y + h

                try:
                    results = self.model.predict(image, bboxes=[x1, y1, x2, y2])
                except Exception as e:
                    print(f"[ERROR] SAM prediction failed for box {x1,y1,x2,y2}: {e}")
                    continue

                if not results or not hasattr(results[0], "masks") or results[0].masks is None:
                    print(f"[WARN] No masks returned for label {label} in box {x1,y1,x2,y2}")
                    continue

                for sam_m in results[0].masks:
                    sam_mask = sam_m.data.cpu().numpy().squeeze().astype(np.uint8)

                    # === DEBUG VISUALIZATION ===
                    debug_overlay = image.copy()
                    debug_overlay[sam_mask == 1] = [255, 0, 0]  # Red: SAM mask
                    cv2.rectangle(debug_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                    plt.figure(figsize=(6, 4))
                    plt.title(f"Label {label} | Box ({x1},{y1})→({x2},{y2})")
                    plt.imshow(cv2.cvtColor(debug_overlay, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()

                    # Assign most common original label inside this SAM mask
                    overlapping_labels = mask[sam_mask == 1]
                    if overlapping_labels.size == 0:
                        continue
                    majority_label = np.bincount(overlapping_labels).argmax()

                    refined_mask[sam_mask == 1] = majority_label

        # Fill in any remaining areas with original mask
        refined_mask[refined_mask == 0] = mask[refined_mask == 0]

        print("[INFO] Refinement complete.")
        return refined_mask
