from ultralytics import YOLO

import cv2

import numpy as np
import matplotlib.pyplot as plt
from roi import get_roi, resize_and_pad

# load trained model
model = YOLO("runs/segment/train13/weights/best.pt")

# train model
# model.train(data='datasets/data.yaml', epochs=3, imgsz=640)

# open test image
img1 = cv2.imread("testImages/IMG_5471copy.jpg")
img2 = get_roi(img1)
# prediction
img = resize_and_pad(img2, 640, 640)
results = model.predict(img)
mask_img = results[0].plot(labels=False, boxes=False)

# Convert the plot image from RGB to BGR format because OpenCV uses BGR
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

masks = results[0].masks
if masks is not None:
    num_masks = len(masks)
    print(f"Number of masks: {num_masks}")

    # Set the position and text color
    text_position = (10, 10)  # Top-left corner of the image
    text_content = f" {num_masks}"
    text_color = (0, 0, 255)  # Red color for the text in BGR format
    font_scale = 1  # Depending on your image size, you may need to adjust this
    thickness = 2  # Thickness of the text

    # Put the text on the image
    cv2.putText(mask_img, text_content, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    for i in range(num_masks):
        mask = masks[i].data[0].numpy()
        mask = (mask * 255).astype(np.uint8)

        # Find the centroid of the mask
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # size of the drawn point
        point_size = 3  # Radius of the drawn point
        # Draw the centroid on the image as a filled circle
        cv2.circle(mask_img, (cX, cY), point_size, (0, 0, 255), -1)  # -1 fills the circle

    # Show the result
    cv2.imshow("result", mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
