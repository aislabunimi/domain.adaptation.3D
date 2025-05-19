import cv2
import numpy as np
from ultralytics import SAM
import matplotlib.pyplot as plt

class SAM2Refiner:
    def __init__(self, model_path: str = "sam2_b.pt"):
        """
        Initializes the SAM2 model.
        """
        self.model = SAM(model_path)

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refines the segmentation mask using SAM2, treating each connected component
        of each class label as a separate instance, skipping wall (1) and floor (2).
        """
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)

        print(f"[INFO] Found labels: {unique_labels}")

        for label in unique_labels:
            if label == 0 or label in [1, 2]:
                continue  # Skip background, wall, and floor

            binary_mask = (mask == label).astype(np.uint8)
            num_components, components = cv2.connectedComponents(binary_mask)

            print(f"[INFO] Label {label} → {num_components - 1} connected components")

            for comp_id in range(1, num_components):  # Skip background
                comp_mask = (components == comp_id).astype(np.uint8)

                # Compute centroid
                moments = cv2.moments(comp_mask)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                try:
                    results = self.model.predict(image, points=[[cx, cy]])
                except Exception as e:
                    print(f"[ERROR] SAM prediction failed at ({cx}, {cy}): {e}")
                    continue

                if not results or not hasattr(results[0], "masks") or results[0].masks is None:
                    print(f"[WARN] No masks returned for label {label} at ({cx}, {cy})")
                    continue

                for sam_m in results[0].masks:
                    sam_mask = sam_m.data.cpu().numpy().squeeze().astype(np.uint8)

                    # === DEBUG VISUALIZATION ===
                    debug_overlay = image.copy()
                    debug_overlay[sam_mask == 1] = [255, 0, 0]  # Red for SAM mask
                    cv2.circle(debug_overlay, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)  # Green dot

                    plt.figure(figsize=(6, 4))
                    plt.title(f"Label {label} | Centroid ({cx}, {cy})")
                    plt.imshow(cv2.cvtColor(debug_overlay, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()

                    # Determine majority label inside this SAM mask
                    overlapping_labels = mask[sam_mask == 1]
                    majority_label = (
                        np.bincount(overlapping_labels).argmax()
                        if overlapping_labels.size > 0
                        else label
                    )

                    refined_mask[sam_mask == 1] = majority_label

        # Fill missing areas with original labels
        refined_mask[refined_mask == 0] = mask[refined_mask == 0]

        print("[INFO] Refinement complete.")
        return refined_mask
