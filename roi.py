import numpy as np
import cv2


def get_roi(image):
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 11)

    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)

    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = adaptive_thresh.shape

    # These will hold the points of the combined bounding box
    x_min = float('inf')
    y_min = float('inf')
    x_max = -float('inf')
    y_max = -float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        area_avg = np.mean(area)

        x, y, w, h = cv2.boundingRect(contour)
        # check if coordinates are touching borders of image
        if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
            continue  # skip since contour touches border of image

        if area_avg * 2 > 200:
            cv2.drawContours(image_copy, [contour], -1, (255, 0, 255), 3)

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    # Draw a rectangle around the combined bounding box
    image_copy_2 = image.copy()

    padding = 50
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)  # Ensure y_min doesn't go below 0
    x_max = min(x_max + padding, image_copy_2.shape[1])  # Ensure x_max doesn't exceed image width
    y_max = min(y_max + padding, image_copy_2.shape[0])

    cv2.rectangle(image_copy_2, (x_min, y_min), (x_max, y_max), (0, 255, 0),
                  2)  # Green rectangle with thickness of 2

    cropped = image_copy_2[y_min:y_max, x_min:x_max]

    return cropped


def resize_and_pad(img, width, height):
    # Calculate the ratio of the new image
    ratio = min(width / img.shape[1], height / img.shape[0])
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)

    # Resize the image with the new ratio
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and black background
    new_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute the center of the new image
    x_center = (width - new_width) // 2
    y_center = (height - new_height) // 2

    # Place the resized image at the center of the new image
    new_img[y_center:y_center + new_height, x_center:x_center + new_width] = resized_img

    return new_img
