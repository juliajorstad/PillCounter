import numpy as np
import cv2
import matplotlib.pyplot as plt

image2 = cv2.imread('pillexample.jpg')
image_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
image_hsl = cv2.cvtColor(image2, cv2.COLOR_BGR2HLS)


# display the image with changed contrast and brightness
cv2.imshow('HSV', image_hsv)
cv2.waitKey(0)

cv2.imshow('HSL', image_hsl)
cv2.waitKey(0)

# preprocessing: gray scale
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
grayHSV = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2GRAY)
grayHSL = cv2.cvtColor(image_hsl, cv2.COLOR_BGR2GRAY)

# preprocessing: blurring
blur = cv2.GaussianBlur(gray,(5,5),0)
blurHSV = cv2.GaussianBlur(grayHSV,(5,5),0)
blurHSL = cv2.GaussianBlur(grayHSL,(5,5),0)

# Convert to binary image with Otsus thresholding
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(blurHSV,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(blurHSV,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('Threshold', th3)
cv2.waitKey(0)

# plot all the images and their histograms

blurred = cv2.GaussianBlur(th3, (9, 9), 5)
edges = cv2.Canny(blurred,20,50)


# WATERSHED

# noise removal with opening. Opening performs erosion and dilatation to remove noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(image2,markers)
image2[markers == -1] = [255,0,0]

jet_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image2, alpha=255.0/dist_transform.max()), cv2.COLORMAP_JET)

# Display the image with jet colormap
cv2.imshow('Jet Colormap', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# get the label of each object
unique_labels = np.unique(markers)

# Exclude the background label (1) and the boundary label (-1)
object_count = len(unique_labels) - 2 # Subtract 2 to exclude background and boundary labels

print("Number of objects detected:", object_count)
image2_copy=image2.copy()
# Iterate through all unique labels, skipping the background (1) and boundaries (-1)
for label in unique_labels:
    if label in [1, -1]:
        continue

    # Create a mask for the current object
    mask = np.zeros_like(markers, dtype=np.uint8)
    mask[markers == label] = 255

    # Find the contours of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the moments of the object
    M = cv2.moments(contours[0])

    # Calculate the centroid of the object
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw a red dot in the middle of the object
    cv2.circle(image2_copy, (cX, cY), 5, (0, 0, 255), -1)

# Display the result
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image2_copy, f'Pills Count: {object_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('Image with Centroids', image2_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -------------------------


