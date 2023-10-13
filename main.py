import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dataset/pill6_cropped.jpg')
cv2.imshow("Original image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def thresholding(img):

    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blurHSV = cv2.GaussianBlur(image_hsv, (5, 5), 0)

    # blur = cv2.medianBlur(img, 5)
    # blurHSV = cv2.medianBlur(grayHSV, 5)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grayHSV = cv2.cvtColor(blurHSV, cv2.COLOR_BGR2GRAY)


    opt_value,otsu_thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    otsu_thresh= cv2.threshold(grayHSV,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]
    cv2.imshow("otsu", otsu_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #plot hisogram
    #plot_histogram(blur,blurHSV,opt_value)


    return otsu_thresh


# WATERSHED
def watershed(th3,img):
    #th3=preprocessing()
    # noise removal with opening. Opening performs erosion and dilatation to remove noise
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th3,cv2.MORPH_OPEN,kernel, iterations = 2)


    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow("sure bg", sure_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    cv2.imshow("sure fg", sure_fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    jet_colormap = cv2.applyColorMap(cv2.convertScaleAbs(dist_transform, alpha=255.0/dist_transform.max()), cv2.COLORMAP_JET)


    cv2.imshow('Image with watershed markers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return markers
def counter(markers):

    # get the label of each object
    unique_labels = np.unique(markers)

    # Exclude the background label (1) and the boundary label (-1)
    object_count = len(unique_labels) - 2 # Subtract 2 to exclude background and boundary labels

    print("Number of objects detected:", object_count)
    image_copy=image.copy()
    # Iterate through all unique labels, skipping the background (1) and boundaries (-1)
    for label in unique_labels:
        if label in [1, -1]:
            continue

        # Create a mask for the current object
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == label] = 255

        # Find the contours of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cv2.contourArea(contours[0]) < 200 or cv2.contourArea(contours[0]) > 10000:
            continue

        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Classify shape based on number of vertices and aspect ratio
        aspect_ratio = w / h
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4 and 0.95 <= aspect_ratio <= 1.05:
            shape = "Square"
        elif vertices == 4:
            shape = "Rectangle"
        elif vertices > 4 and 0.88 <= aspect_ratio <= 1.12:
            shape = "Circle"
        else:
            shape = "Ellipse"


        # Calculate the moments of the object
        M = cv2.moments(contours[0])

        if M["m00"] != 0:
            # Calculate the centroid of the object
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Handle the zero division error
            cX, cY = 0, 0  # or any other appropriate value

        # Draw a red dot in the middle of the object
        cv2.circle(image_copy, (cX, cY), 10, (0, 0, 255), -1)
        cv2.putText(image_copy, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_copy, f'Pills Count: {object_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image with Centroids', image_copy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def plot_histogram(blur,blurHSV,ret3):
    # Plot the histogram of Otsu's threshold
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(blur.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.axvline(x=ret3, color='red', linestyle='dashed', linewidth=2)
    plt.title("Histogram of Grayscale Image with Otsu's Threshold")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(blurHSV.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.axvline(x=ret3, color='red', linestyle='dashed', linewidth=2)
    plt.title("Histogram of HSV Image with Otsu's Threshold")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()



thresh=thresholding(image)
markers=watershed(thresh,image)
counter(markers)

