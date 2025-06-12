import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class InnerBoxFinder:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def find_largest_inner_box(self, mask: np.ndarray):
        """
        Finds the largest axis-aligned rectangle fully inside the binary mask.

        Parameters:
            mask (np.ndarray): Binary mask of shape (H, W)

        Returns:
            tuple: ((cx, cy), (x, y, w, h)) - centroid and rectangle coordinates,
                   or (None, None) if no valid rectangle found.
        """
        if mask.ndim != 2:
            raise ValueError("Input mask must be a 2D binary array.")

        height, width = mask.shape
        heights = [0] * width
        max_area = 0
        best_rect = None

        for i in range(height):
            # Update histogram of consecutive '1's column-wise
            for j in range(width):
                heights[j] = heights[j] + 1 if mask[i, j] else 0

            # Use stack-based histogram approach to find max rectangle in row
            stack = []
            for j in range(width + 1):
                current_height = heights[j] if j < width else 0
                start = j
                while stack and current_height < stack[-1][1]:
                    prev_start, prev_height = stack.pop()
                    x0 = prev_start
                    y0 = i - prev_height + 1
                    w0 = j - prev_start
                    h0 = prev_height

                    if w0 > 0 and h0 > 0:
                        rect_region = mask[y0:y0 + h0, x0:x0 + w0]
                        # Ensure the entire region is inside the mask
                        if rect_region.shape == (h0, w0) and np.all(rect_region == 1):
                            area = w0 * h0
                            if area > max_area:
                                max_area = area
                                best_rect = (x0, y0, w0, h0)
                    start = prev_start
                stack.append((start, current_height))

        if best_rect is None:
            return None, None

        x, y, w, h = best_rect
        cx = x + w // 2
        cy = y + h // 2

        if self.debug:
            self._visualize(mask, best_rect)

        return (cx, cy), best_rect

    def _visualize(self, mask: np.ndarray, rect: tuple):
        """
        Visualizes the mask and overlays the largest inner rectangle.

        Parameters:
            mask (np.ndarray): Binary mask
            rect (tuple): (x, y, w, h) of the rectangle
        """
        x, y, w, h = rect

        fig, ax = plt.subplots()
        ax.imshow(mask, cmap='gray')
        rect_patch = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect_patch)
        ax.set_title("Largest Valid Inner Rectangle")
        plt.show()
