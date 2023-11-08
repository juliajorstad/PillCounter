import cv2
import numpy as np

# CREATES A TRACKBAR TO ADJUST THRESHOLD IN REAL TIME
def update_threshold(x):
    # Retrieve the current trackbar values
    blockSize = cv2.getTrackbarPos("Block Size", "Adaptive Thresholding")
    C = cv2.getTrackbarPos("Constant (C)", "Adaptive Thresholding")

    # Ensure that blockSize is odd and greater than 1
    blockSize = max(3, blockSize)  # Minimum value of 3 to ensure odd number
    blockSize = blockSize + 1 if blockSize % 2 == 0 else blockSize  # Make it odd

    # Print the current values
    print(f"Block Size: {blockSize}, C: {C}")

    # Apply adaptive thresholding using the updated values
    thresh_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C
    )

    # Display the current values on a separate text image
    text_image = thresh_img.copy()
    cv2.putText(
        text_image,
        f"Block Size: {blockSize}, C: {C}",
        (10, 30),  # Position of the text
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,  # Font scale
        (255, 255, 255),  # Text color (white)
        2,  # Thickness of the text
        cv2.LINE_AA,  # Line type
    )

    # Combine the thresholded image with the text image
    combined_image = cv2.addWeighted(thresh_img, 0.7, text_image, 0.3, 0)

    cv2.imshow("Adaptive Thresholding", combined_image)

# Load an image (replace 'your_image.jpg' with your image path)
image = cv2.imread('datasets/pill19.jpg')

# Convert the image to grayscale
d = 9  # Diameter of the pixel neighborhood
sigma_color = 75  # Color sigma (larger values preserve more colors)
sigma_space = 75  # Space sigma (larger values preserve more structure)

# Apply Bilateral Filter
blur = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Create a window to display the results
cv2.namedWindow("Adaptive Thresholding")

# Initialize trackbar values
blockSize = 11  # Initial blockSize value (should be an odd number)
C = 2  # Initial constant (C) value

# Create trackbars for blockSize and C
cv2.createTrackbar("Block Size", "Adaptive Thresholding", blockSize, 255, update_threshold)
cv2.createTrackbar("Constant (C)", "Adaptive Thresholding", C, 255, update_threshold)

# Initial adaptive thresholding
update_threshold(0)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' key to exit
        break

cv2.destroyAllWindows()