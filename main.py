import numpy as np
import cv2
import matplotlib.pyplot as plt

image2 = cv2.imread('pillexample.jpg')
image = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV )
image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2HLS )

# display the image with changed contrast and brightness
cv2.imwrite('pill_HSV.jpg', image)
cv2.imshow('HSV', image)
cv2.waitKey(0)

cv2.imwrite('pill_HSL.jpg', image3)
cv2.imshow('HSL', image3)
cv2.waitKey(0)

gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imwrite('pill_gray.jpg', gray)
cv2.imshow('gray', gray)
cv2.waitKey(0)

grayHSV = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('pill_grayHSV.jpg', grayHSV)
cv2.imshow('grayHSV', grayHSV)
cv2.waitKey(0)

grayHSL = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
cv2.imwrite('pill_grayHSL.jpg', grayHSL)
cv2.imshow('grayHSL', grayHSL)
cv2.waitKey(0)

blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imwrite('pill_black_and_white.jpg', th3)
cv2.imshow('None approximation', th3)
cv2.waitKey(0)

blurHSV = cv2.GaussianBlur(grayHSV,(5,5),0)
ret3,th3 = cv2.threshold(blurHSV,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imwrite('pill_black_and_whiteHSV.jpg', th3)
cv2.imshow('None approximation', th3)
cv2.waitKey(0)
# plot all the images and their histograms

blurHSL = cv2.GaussianBlur(grayHSL,(5,5),0)
ret3,th3 = cv2.threshold(blurHSV,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imwrite('pill_black_and_whiteHSL.jpg', th3)
cv2.imshow('None approximation', th3)
cv2.waitKey(0)
# plot all the images and their histograms

blurred = cv2.GaussianBlur(th3, (9, 9), 5)
edges = cv2.Canny(blurred,20,50)

cv2.imshow('None approximation', edges)
cv2.waitKey(0)





ret, thresh = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)

cv2.imshow('None approximation', thresh)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)

for contour in contours:
    # Filter based on area to remove small noise
    if cv2.contourArea(contour) > 10:  # You might need to adjust this threshold based on your image
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
cv2.putText(image, f'Pills Count: {count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('Detected Pills', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

