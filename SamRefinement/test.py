
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM

# === Load and prepare image ===
image_bgr = cv2.imread("test_db/color/618.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (320, 240), interpolation=cv2.INTER_AREA)

# === Load SAM model ===
model = SAM("sam2_b.pt")

# === Click callback ===
def on_click(event):
    if event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)
    print(f"Clicked at: ({x}, {y})")

    # Run SAM with the clicked point
    results = model.predict(image_rgb, points=[[x, y]])

    if not results or not results[0].masks:
        print("No mask returned.")
        return

    # Get first mask
    sam_mask = results[0].masks[0].data.cpu().numpy().squeeze().astype(np.uint8)

    # Display result
    overlay = image_rgb.copy()
    overlay[sam_mask == 1] = [255, 0, 0]  # Red mask overlay

    ax.clear()
    ax.imshow(overlay)
    ax.set_title(f"SAM Segmentation at ({x}, {y})")
    plt.draw()

# === Show image and connect click handler ===
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image_rgb)
ax.set_title("Click to run SAM")
fig.canvas.mpl_connect('button_press_event', on_click)
plt.axis('off')
plt.show()
