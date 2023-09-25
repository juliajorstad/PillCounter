import numpy as np
import cv2

image = cv2.imread('pillexample2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)



ret, thresh = cv2.threshold(blurred,230,255,cv2.THRESH_BINARY)

# Erode to separate objects that are close to each other
kernel = np.ones((7,7),np.uint8)
eroded = cv2.erode(thresh, kernel, iterations = 1)

cv2.imshow('None approximation', eroded)
cv2.waitKey(0)


contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image_copy,contours,-1,(0, 255, 0),2,
                 cv2.LINE_AA)

# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

# Simplified counting process
count = sum(1 for contour in contours if cv2.contourArea(contour) > 40)

# Display the count on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, f'Pill Count: {count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('Detected Pills', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




