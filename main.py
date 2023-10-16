import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dataset/pill13.jpg')
cv2.imshow("Original image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
def thresholding():


    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Gaussian blur
    #blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # median blur
    blur = cv2.medianBlur(gray, 11)
    # bilateral blur
    #blur= cv2.bilateralFilter(gray, 7, 75, 55)
    cv2.imshow("gaussian blur", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    adaptive_thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,2)

    cv2.imshow("threshold", adaptive_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    open = cv2.morphologyEx(adaptive_thresh,cv2.MORPH_OPEN,(3,3))
    cv2.imshow("OPEN", open)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(adaptive_thresh, kernel)

    #cv2.imshow("dilate", dilate)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()





    #dilate = cv2.dilate(canny, (7, 7), iterations=2)
    #cv2.imshow("dilate", dilate)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # binary inv if the pills are darker than background, use binary thresh if lighter


    return adaptive_thresh

def get_roi(adaptive_thresh):
    image_copy = image.copy()

    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height,width=adaptive_thresh.shape

    # These will hold the points of the combined bounding box
    x_min = float('inf')
    y_min = float('inf')
    x_max = -float('inf')
    y_max = -float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        area_avg = np.mean(area)

        x, y, w, h = cv2.boundingRect(contour)


        if x ==0 or y==0 or (x+w) == width or (y+h)==height:
            continue # skip since contour touches border of image

        if area_avg*2> 200:
            cv2.drawContours(image_copy, [contour], -1, (255, 0, 255), 3)

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    cv2.imshow("contours", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw a rectangle around the combined bounding box
    image_copy_2 = image.copy()

    padding=50
    x_min=max(x_min - padding,0)
    y_min = max(y_min - padding, 0)  # Ensure y_min doesn't go below 0
    x_max = min(x_max + padding, image_copy_2.shape[1])  # Ensure x_max doesn't exceed image width
    y_max = min(y_max + padding, image_copy_2.shape[0])

    cv2.rectangle(image_copy_2, (x_min, y_min), (x_max, y_max), (0, 255, 0),
                         2)  # Green rectangle with thickness of 2
    bbox=(x_min,y_min,x_max,y_max)

    # Show the image with bounding box
    cv2.imshow("Region of interest", image_copy_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return bbox

def apply_threshold_in_bbox(image, bbox):
    # Extract the ROI using the bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    roi = image[y_min:y_max, x_min:x_max]


    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # median blur
    #blur = cv2.medianBlur(gray, 3)
    thresh_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
    cv2.imshow('thres_gray', thresh_gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cnts, hier = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    res = np.zeros_like(gray)
    cv2.drawContours(res, [c], -1, 255, -1)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Use floodFill for filling the center of the contour
    cv2.floodFill(res, None, (cX, cY), 255)

    # Show images for testing
    cv2.imshow('res', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #canny = cv2.Canny(roi, 0, 40)

    #cv2.imshow("canny", blur)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, (11,11))


    return image
# WATERSHED
def watershed(thresh, img):
    # noise removal with opening. Opening performs erosion and dilatation to remove noise

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    cv2.imshow("morph", opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow("sure bg", sure_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    #ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)

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
    pills = []
    # Exclude the background label (1) and the boundary label (-1)
    object_count = len(unique_labels) - 2 # Subtract 2 to exclude background and boundary labels

    print("Number of objects detected:", object_count)
    image_copy=image.copy()
    # Iterate through all unique labels, skipping the background (1) and boundaries (-1)
    for label in unique_labels[2:]:

        # Create a mask for the current object
        target = np.where(markers == label,255,0).astype(np.uint8)

        # Find the contours of the object
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pills.append(contours[0])

        """if cv2.contourArea(contours[0]) < 200 or cv2.contourArea(contours[0]) > 10000:
            continue"""

        # Approximate the contour
        perimeter = 0.02 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], perimeter, True)

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
        #cv2.drawContours(image_copy,pills,-1,color=(0,23,223), thickness=5)

    # Display the result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_copy, f'Pills Count: {object_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #image_copy = cv2.drawContours(image_copy,pills,-1,color=(0,23,223), thickness=5)
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



thresh=thresholding()
bbox=get_roi(thresh)
#apply_threshold_in_bbox(image,bbox)
#markers=watershed(thresh,image)
#counter(markers)