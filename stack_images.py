import cv2
import numpy as np


def stack_images_vertically(image_list):
    # Calculate the width and height of the output image
    width = max(img.shape[1] for img in image_list)
    height = sum(img.shape[0] for img in image_list)

    # Determine the number of channels for the output image
    num_channels = image_list[0].shape[2] if len(image_list[0].shape) == 3 else 1

    # Create an empty canvas for the stacked image
    stacked_image = np.zeros((height, width, num_channels), dtype=np.uint8)

    # Initialize the y-coordinate to keep track of the vertical position
    y_offset = 0

    # Iterate through the images and stack them vertically
    for img in image_list:
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel grayscale
        h, w, _ = img.shape
        stacked_image[y_offset:y_offset + h, 0:w] = img
        y_offset += h

    return stacked_image
