import cv2
import numpy as np


import math
import matplotlib.pyplot as plt

image = cv2.imread('dataset/pill9.jpg')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (7, 7), 0)

canny =cv2.Canny(blurred,30,150,3)

diltated=cv2.dilate(canny,(1,1), iterations=0)



contours, hierarchy = cv2.findContours(diltated.copy(),
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


# draw contours on the original image

cv2.drawContours(rgb,contours,-1,(0, 255, 0),2)


for contour in contours:
    # Filter based on area to remove small noise
    if cv2.contourArea(contour) > 40:  # You might need to adjust this threshold based on your image
        M = cv2.moments(contour)

        # Calculate the center of the contour (centroid)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw a red dot in the center
        cv2.circle(rgb, (cX, cY), 5, (0, 0, 255), -1)

# Simplified counting process
count = sum(1 for contour in contours if cv2.contourArea(contour) > 40)

# Display the count on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(rgb, f'Pills Count: {count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('Detected Pills', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


