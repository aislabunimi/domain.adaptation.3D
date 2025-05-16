import cv2
import numpy as np

# Path to your image file
image_path = '/media/adaptation/New_volume/Domain_Adaptation_Pipeline/IO_pipeline/Scannet_DB/scans/scene0000_00/deeplab_labels/10.png'

# Read the image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read the image as is

# Check the number of channels in the image
if image is not None:
    print("Image shape:", image.shape)  # This will output (height, width) or (height, width, channels)

    # If there are 1 channel, it's mono (grayscale)
    if len(image.shape) == 2:
        print("The image is MONO (grayscale).")
    elif len(image.shape) == 3 and image.shape[2] == 4:
        print("The image is RGBA (has 4 channels).")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        print("The image is RGB (has 3 channels).")
    else:
        print("Unknown image format.")

    # Compute and print min and max pixel values
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    print(f"Minimum pixel value: {min_pixel}")
    print(f"Maximum pixel value: {max_pixel}")

else:
    print("Error: Image could not be loaded.")
