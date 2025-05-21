import cv2
import numpy as np
from ultralytics.models.sam import SAM2Predictor
import matplotlib.pyplot as plt


class SAM2RefinerMixed:
    def __init__(self, model_path: str = "sam2_b.pt"):
        overrides = dict(conf=0.25, task="segment", mode="predict",verbose=False, imgsz=1024, model=model_path)
        self.predictor = SAM2Predictor(overrides=overrides)

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        refined_mask = mask.copy()
        unique_labels = np.unique(mask)
        print(f"[INFO] Found labels: {unique_labels}")

        for label in unique_labels:
            if label == 0:
                continue

            original_mask = (mask == label).astype(np.uint8)

            # Dilate to merge nearby fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(original_mask, kernel, iterations=1)

            # Extract connected components from dilated mask
            num_components, components = cv2.connectedComponents(dilated_mask)
            print(f"[INFO] Label {label} → {num_components - 1} components")

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)

                

                # Bounding box from valid component mask
                x, y, w, h = cv2.boundingRect(comp_mask)
                box = [x, y, x + w, y + h]
                ys, xs = np.where(comp_mask == 1)
                # Compute centroid from valid component mask
                M = cv2.moments(comp_mask)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # If centroid is outside, snap it to nearest foreground pixel
                cx, cy = np.median(xs).astype(int), np.median(ys).astype(int)
                prompt_points = [[cx, cy]]

                try:
                    results = self.predictor(
                        source=[image],
                        points=[prompt_points],
                        #labels=[prompt_labels],
                        boxes=[[box]]
                    )
                except Exception as e:
                    print(f"[ERROR] SAM failed for label {label} @ {box}: {e}")
                    continue

                if not results or results[0].masks is None:
                    print(f"[WARN] No mask returned for label {label}")
                    continue

                sam_mask = results[0].masks[0].data.cpu().numpy().squeeze().astype(np.uint8)

                # Determine dominant pseudo label in returned mask
                overlap_labels = mask[sam_mask == 1]
                if overlap_labels.size == 0:
                    continue
                dominant_label = np.bincount(overlap_labels).argmax()

                refined_mask[sam_mask == 1] = dominant_label

                # === Debug visualization ===
                debug = image.copy()
                debug[sam_mask == 1] = [255, 0, 0]  # Red for prediction
                cv2.circle(debug, (cx, cy), 4, (0, 255, 0), -1)  # Green dot for prompt point
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)

                plt.figure(figsize=(6, 4))
                plt.title(f"Label {label} → Dominant: {dominant_label} | Centroid: ({cx},{cy})")
                plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()

        return refined_mask
