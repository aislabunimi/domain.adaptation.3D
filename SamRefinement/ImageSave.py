import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM
import os


class SAM2RefinerFast:
    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        save_dir: str = "refined_outputs",
        skip_labels: list = None,
        skip_max_labels: list = None,
        batch_size: int = 8,
        fill_strategy: str = "ereditary",
        min_area_ratio: float = None
    ):
        """
        Args:
            model_path: Path to the SAM model checkpoint.
            save_dir: Directory where visual outputs will be saved.
            skip_labels: Labels to completely ignore.
            skip_max_labels: Labels to assign SAM output directly (skip majority voting).
            batch_size: Batch size for SAM predictions.
            fill_strategy: "maxlabel" (most common label) or "ereditary" (fallback to original mask).
            min_area_ratio: Minimum component area (relative to image) to consider.
        """
        self.model = SAM(model_path)
        self.skip_labels = skip_labels if skip_labels else []
        self.skip_max_labels = skip_max_labels if skip_max_labels else []
        self.batch_size = batch_size
        assert fill_strategy in ("maxlabel", "ereditary"), "Invalid fill strategy"
        self.fill_strategy = fill_strategy
        self.min_area_ratio = min_area_ratio
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def refine(self, image: np.ndarray, mask: np.ndarray, filename_prefix: str = "refined") -> np.ndarray:
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        unique_labels = np.unique(mask)
        image_area = image.shape[0] * image.shape[1]

        all_points, all_bboxes, prompt_labels, comp_masks = [], [], [], []

        for label in unique_labels:
            if label == 0 or label in self.skip_labels:
                continue

            binary_mask = (mask == label).astype(np.uint8)
            dilated = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            num_components, components = cv2.connectedComponents(dilated)

            for comp_id in range(1, num_components):
                comp_mask = (components == comp_id).astype(np.uint8)
                area = np.sum(comp_mask)
                if self.min_area_ratio and area < image_area * self.min_area_ratio:
                    continue

                ys, xs = np.where(comp_mask == 1)
                if len(xs) == 0:
                    continue

                cx, cy = int(np.median(xs)), int(np.median(ys))
                x, y, w, h = cv2.boundingRect(comp_mask)
                all_points.append([cx, cy])
                all_bboxes.append([x, y, x + w, y + h])
                prompt_labels.append(label)
                comp_masks.append(comp_mask)

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
                if labels_batch[j] in self.skip_max_labels:
                    refined_mask[sam_mask == 1] = labels_batch[j]
                else:
                    overlapping_labels = mask[sam_mask == 1]
                    if overlapping_labels.size == 0:
                        continue
                    majority_label = np.bincount(overlapping_labels).argmax()
                    refined_mask[sam_mask == 1] = majority_label

        # Fill unassigned
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
        else:
            refined_mask[to_fill] = mask[to_fill]

        self._save_visualizations(image, refined_mask, all_points, all_bboxes, filename_prefix)
        return refined_mask
    def _save_visualizations(self, image, mask, points, bboxes, output_prefix="refined"):
        def create_label_colormap():
            np.random.seed(42)
            colormap = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
            colormap[0] = [0, 0, 0]
            return colormap

        colormap = create_label_colormap()
        color_mask = colormap[mask]
        blended = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)

        def draw_plot(base_img, output_path):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            ax.imshow(base_img)
            ax.axis("off")

            np.random.seed(123)  # Ensure reproducibility
            for idx, ((x, y), (x1, y1, x2, y2)) in enumerate(zip(points, bboxes)):
                # Draw red point
                ax.plot(x, y, 'ro', markersize=6)

                # Add label
                ax.text(x + 3, y - 3, f"#{idx}", color='white', fontsize=6,
                        bbox=dict(facecolor='black', edgecolor='none', pad=1))

                # Unique box color
                color = np.random.rand(3,)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=3, edgecolor=color, facecolor='none',
                                    linestyle='dashed')
                ax.add_patch(rect)

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        draw_plot(image, f"{output_prefix}_rgb.png")
        draw_plot(blended, f"{output_prefix}_masked.png")