import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
from ultralytics import SAM


class FastSamRefinerAuto:
    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        visualize: bool = True,
        granularity: int = 32,
    ):
        """
        Args:
            model_path: Path to the SAM model checkpoint.
            visualize: Whether to show debug plots and save outputs.
            granularity: Distance in pixels between grid points (lower is finer).
        """
        self.model = SAM(model_path)
        self.visualize = visualize
        self.granularity = granularity

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        refined_mask = np.zeros_like(mask, dtype=np.uint8)

        # Step 1: Generate grid points
        h, w = image.shape[:2]
        ys = np.arange(0, h, self.granularity)
        xs = np.arange(0, w, self.granularity)
        grid_points = np.array([[x, y] for y in ys for x in xs])

        if len(grid_points) == 0:
            print("[WARN] No grid points generated.")
            return mask.copy()

        # Save debug grid points
        if self.visualize:
            self._save_grid_debug(image, grid_points)
        return
        try:
            results = self.model.predict(image, points=grid_points, labels=[1] * len(grid_points), verbose=False)
        except Exception as e:
            print(f"[ERROR] SAM segmentation with prompts failed: {e}")
            return mask.copy()

        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            print("[WARN] No masks found by SAM.")
            return mask.copy()

        sam_masks = results[0].masks.data.cpu().numpy().astype(np.uint8)

        # Sort masks by area
        areas = np.sum(sam_masks, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        sam_masks = sam_masks[sorted_indices]

        if self.visualize:
            vis_image = image.copy()

        for idx, sam_mask in enumerate(sam_masks):
            overlapping_labels = mask[sam_mask == 1]
            valid_labels = overlapping_labels[overlapping_labels != 0]

            if valid_labels.size == 0:
                continue

            majority_label = np.bincount(valid_labels).argmax()

            apply_mask = (sam_mask == 1) & (refined_mask == 0)
            refined_mask[apply_mask] = majority_label

            if self.visualize:
                vis_image = self._draw_mask_on_image(
                    vis_image, sam_mask, majority_label, idx
                )

        # Step 2: Fill remaining regions
        unassigned = (refined_mask == 0) & (mask != 0)
        refined_mask[unassigned] = mask[unassigned]

        # Step 3: Save final overlay
        if self.visualize:
            self._save_final_overlay(image, refined_mask)

        return refined_mask

    def _save_grid_debug(self, image: np.ndarray, points: np.ndarray, path="sam_grid_debug.png"):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.imshow(image)
        for idx, (x, y) in enumerate(points):
            # Draw red point
            ax.plot(x, y, 'ro', markersize=10)

            # Add label
            ax.text(x + 3, y + 3, f"#{idx}", color='white', fontsize=6,
                    bbox=dict(facecolor='black', edgecolor='none', pad=1))

                
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _draw_mask_on_image(self, image, mask, label, idx):
        color_map = get_cmap("tab20")
        color = color_map(idx % 20)[:3]  # RGB tuple
        color_bgr = tuple(int(c * 255) for c in reversed(color))  # BGR for cv2

        # Create transparent overlay
        overlay = image.copy()


        # Get bounding box
        ys, xs = np.where(mask == 1)
        if ys.size == 0 or xs.size == 0:
            return image

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color + (0.3,),
            linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(
            x0, y0 - 4, f"Label {label}", color='white', fontsize=8,
            bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8)
        )
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _save_final_overlay(self, image, mask, path="sam_final_overlay.png"):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.imshow(image)
        ax.imshow(mask, cmap='jet', alpha=0.5)
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
