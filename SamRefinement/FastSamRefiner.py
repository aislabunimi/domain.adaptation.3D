import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM
from InnerBoxFinder import InnerBoxFinder

class SAM2RefinerMixed:
    def __init__(self, model_path="sam2_b.pt", visualize=True, skip_labels=None,
                 batch_size=8, fill_strategy="maxlabel"):
        self.model = SAM(model_path)
        self.visualize = visualize
        self.skip_labels = skip_labels or []
        self.batch_size = batch_size
        assert fill_strategy in ("maxlabel", "ereditary"), "Invalid fill strategy"
        self.fill_strategy = fill_strategy

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)

        all_points = []
        all_bboxes = []
        prompt_labels = []
        all_rects = []

        finder = InnerBoxFinder(debug=False)

        for label in unique_labels:
            if label == 0 or label in self.skip_labels:
                continue

            binary_mask = (mask == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            num_components, components = cv2.connectedComponents(dilated_mask)

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)
                ys, xs = np.where(comp_mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                                # Crop original and component masks to the bounding box
                cropped_original_mask = binary_mask[y0:y1+1, x0:x1+1]
                cropped_comp_mask = comp_mask[y0:y1+1, x0:x1+1]

                # Intersection: keep only areas present in both
                intersection_mask = np.logical_and(cropped_original_mask, cropped_comp_mask).astype(np.uint8)

                # Find centroid and inner box in intersection only
                centroid, rect = finder.find_largest_inner_box(intersection_mask)

                if centroid is not None:
                    cx, cy = centroid[0] + x0, centroid[1] + y0
                    rect = (rect[0] + x0, rect[1] + y0, rect[2], rect[3])
                else:
                    # Fallback to median of points in the intersection
                    ys_int, xs_int = np.where(intersection_mask)
                    if len(xs_int) == 0 or len(ys_int) == 0:
                        continue  # Skip if no valid intersection
                    cx, cy = int(np.median(xs_int)) + x0, int(np.median(ys_int)) + y0
                    rect = None

                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(comp_mask)
                bbox = [x_bb, y_bb, x_bb + w_bb, y_bb + h_bb]

                all_points.append([cx, cy])
                all_bboxes.append(bbox)
                all_rects.append(rect)
                prompt_labels.append(label)

        if not all_points:
            print("[WARN] No valid prompts found.")
            return mask.copy()

        for i in range(0, len(all_points), self.batch_size):
            points_batch = all_points[i:i + self.batch_size]
            bboxes_batch = all_bboxes[i:i + self.batch_size]
            labels_batch = prompt_labels[i:i + self.batch_size]

            try:
                results = self.model.predict(
                    image,
                    points=points_batch,
                    bboxes=bboxes_batch,
                    verbose=False
                )
            except Exception as e:
                print(f"[ERROR] SAM prediction failed on batch {i}-{i + self.batch_size}: {e}")
                continue

            if not results or not hasattr(results[0], "masks") or results[0].masks is None:
                continue

            for j, sam_m in enumerate(results[0].masks):
                sam_mask = sam_m.data.cpu().numpy().squeeze().astype(np.uint8)
                overlapping_labels = mask[sam_mask == 1]
                if overlapping_labels.size == 0:
                    continue
                majority_label = np.bincount(overlapping_labels).argmax()
                refined_mask[sam_mask == 1] = majority_label

        # Fill unassigned pixels
        originally_labeled = (mask != 0)
        unassigned = (refined_mask == 0)
        to_fill = unassigned & originally_labeled

        if self.fill_strategy == "maxlabel":
            num_components, comp_map = cv2.connectedComponents(to_fill.astype(np.uint8))
            for comp_id in range(1, num_components):
                comp_mask = (comp_map == comp_id)
                original_labels = mask[comp_mask]
                valid_labels = original_labels[original_labels != 0]
                if valid_labels.size == 0:
                    continue
                majority_label = np.bincount(valid_labels).argmax()
                refined_mask[comp_mask] = majority_label
        else:  # eredetary
            refined_mask[to_fill] = mask[to_fill]

        if self.visualize:
            self._debug_visualize(image, refined_mask, all_points, rects=None, bboxes=all_bboxes)

        return refined_mask

    def _debug_visualize(self, image, mask, points=None, rects=None, bboxes=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(image)

        if points:
            for point in points:
                if point:
                    x, y = point
                    plt.plot(x, y, 'ro', markersize=3)

        if rects:
            for rect in rects:
                if rect:
                    x, y, w, h = rect
                    plt.gca().add_patch(
                        plt.Rectangle((x, y), w, h, edgecolor='cyan', facecolor='none', linewidth=1)
                    )

        if bboxes:
            for bbox in bboxes:
                if bbox:
                    x1, y1, x2, y2 = bbox
                    plt.gca().add_patch(
                        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', linewidth=1)
                    )

        plt.imshow(mask, alpha=0.4, cmap='jet')
        plt.title("Refinement Debug View")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
