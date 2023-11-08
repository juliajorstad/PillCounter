import cv2
import numpy as np


import math
import matplotlib.pyplot as plt


def get_contours (img, img_copy):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:
            count +=1
            cv2.drawContours(img_copy, cnt, -1, (255, 0, 255), 3)
            peri = cv2.arcLength(cnt,True) # True: contour is closed
            approx = cv2.approxPolyDP(cnt,0.01* peri, True) # approx shape, gives points
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_copy,(x,y),(x+w,y+h), (0,255,0),3)

    cv2.putText(img_copy, f"Objects: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("contour detection", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('datasets/pill19.jpg')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (7,7), 0)

canny =cv2.Canny(blurred,5,80)
kernel = np.ones((5,5))
diltated=cv2.dilate(canny,kernel, iterations=1)


cv2.imshow("edge detection",diltated)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_copy=image.copy()

get_contours(diltated,img_copy)



