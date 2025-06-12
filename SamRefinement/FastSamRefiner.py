import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

class SAM2RefinerFast:
    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        visualize: bool = True,
        skip_labels: list = None,
        skip_max_labels: list = None,
        batch_size: int = 8,
        fill_strategy: str = "ereditary",
        min_area_ratio: float = None
    ):
        """
        Args:
            model_path: Path to the SAM model checkpoint.
            visualize: Whether to show debug plots.
            skip_labels: Labels to completely ignore.
            skip_max_labels: Labels to assign SAM output directly (skip majority voting).
            batch_size: Batch size for SAM predictions.
            fill_strategy: "maxlabel" (most common label) or "ereditary" (fallback to original mask).
            min_area_ratio: Minimum component area (relative to image) to consider.
        """
        self.model = SAM(model_path)
        self.visualize = visualize
        self.skip_labels = skip_labels if skip_labels else []
        self.skip_max_labels = skip_max_labels if skip_max_labels else []
        self.batch_size = batch_size
        assert fill_strategy in ("maxlabel", "ereditary"), "Invalid fill strategy"
        self.fill_strategy = fill_strategy
        self.min_area_ratio = min_area_ratio

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)
        image_area = image.shape[0] * image.shape[1]

        all_points = []
        all_point_labels = []
        all_bboxes = []
        prompt_labels = []
        comp_masks = []

        # Step 1: Collect prompts
        for label in unique_labels:
            if label == 0 or label in self.skip_labels:
                continue

            binary_mask = (mask == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(binary_mask, kernel, iterations=1)
            num_components, components = cv2.connectedComponents(dilated)

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)
                area = np.sum(comp_mask)
                if self.min_area_ratio is not None and area < image_area * self.min_area_ratio:
                    continue

                ys, xs = np.where(comp_mask == 1)
                if len(xs) == 0:
                    continue

                # Positive points: center + random samples from mask
                pos_points = []
                cx, cy = int(np.median(xs)), int(np.median(ys))
                pos_points.append([cx, cy])

                if len(xs) > 5:
                    indices = np.random.choice(len(xs), size=min(4, len(xs)), replace=False)
                    for idx in indices:
                        pos_points.append([xs[idx], ys[idx]])

                # Optional: add negative points around the bounding box (not in current label)
                neg_points = []
                expanded_mask = cv2.dilate(comp_mask, kernel, iterations=2)
                border_mask = (expanded_mask == 1) & (comp_mask == 0) & (mask != label)
                neg_ys, neg_xs = np.where(border_mask)
                if len(neg_xs) > 0:
                    neg_indices = np.random.choice(len(neg_xs), size=min(4, len(neg_xs)), replace=False)
                    for idx in neg_indices:
                        neg_points.append([neg_xs[idx], neg_ys[idx]])

                all_points.append(pos_points + neg_points)
                all_point_labels.append([1] * len(pos_points) + [-1] * len(neg_points))

                x, y, w, h = cv2.boundingRect(comp_mask)
                all_bboxes.append([x, y, x + w, y + h])
                prompt_labels.append(label)
                comp_masks.append(comp_mask)

        if not all_points:
            print("[WARN] No valid prompts found.")
            return mask.copy()

        # Step 2: SAM predictions in batches
        for i in range(0, len(all_points), self.batch_size):
            points_batch = all_points[i:i + self.batch_size]
            labels_batch = all_point_labels[i:i + self.batch_size]
            bboxes_batch = all_bboxes[i:i + self.batch_size]
            prompt_labels_batch = prompt_labels[i:i + self.batch_size]
            comps_batch = comp_masks[i:i + self.batch_size]

            try:
                results = self.model.predict(
                    image,
                    points=points_batch,
                    labels=labels_batch,
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

                if prompt_labels_batch[j] in self.skip_max_labels:
                    refined_mask[sam_mask == 1] = prompt_labels_batch[j]
                else:
                    overlapping_labels = mask[sam_mask == 1]
                    if overlapping_labels.size == 0:
                        continue
                    majority_label = np.bincount(overlapping_labels).argmax()
                    refined_mask[sam_mask == 1] = majority_label

                if self.visualize:
                    self._debug_per_prompt(
                        image,
                        comps_batch[j],
                        points_batch[j],
                        bboxes_batch[j],
                        sam_mask
                    )

        # Step 3: Fill unassigned areas
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

        elif self.fill_strategy == "ereditary":
            refined_mask[to_fill] = mask[to_fill]

        if self.visualize:
            self._debug_visualize(image, refined_mask, all_points, all_bboxes)

        return refined_mask

    def _debug_per_prompt(self, image, comp_mask, points, bbox, sam_mask):
        x1, y1, x2, y2 = bbox

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # === Plot on Component Mask ===
        axes[0].imshow(comp_mask, cmap='gray')
        for cx, cy in points:
            axes[0].plot(cx, cy, 'ro')
        axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor='lime', facecolor='none', linewidth=1))
        axes[0].set_title("Original Component")
        axes[0].axis('off')

        # === Plot on Original Image ===
        axes[1].imshow(image)
        for cx, cy in points:
            axes[1].plot(cx, cy, 'ro')
        axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor='lime', facecolor='none', linewidth=1))
        axes[1].set_title("Prompt on Image")
        axes[1].axis('off')

        # === SAM Output ===
        axes[2].imshow(sam_mask, cmap='gray')
        axes[2].set_title("SAM Output")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    def _debug_visualize(self, image, mask, points, bboxes):
        def create_label_colormap():
            # Generate 256 distinct colors
            np.random.seed(42)  # Fixed seed for reproducibility
            colormap = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
            colormap[0] = [0, 0, 0]  # Background stays black
            return colormap

        colormap = create_label_colormap()
        color_mask = colormap[mask]

        # Blend original image with color mask
        blended = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)

        plt.figure(figsize=(10, 8))
        plt.imshow(blended)
        for (x, y) in points:
            plt.plot(x, y, 'ro', markersize=3)
        for (x1, y1, x2, y2) in bboxes:
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            edgecolor='lime', facecolor='none', linewidth=1))
        plt.title("Refined Mask Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
