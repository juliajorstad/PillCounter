from ultralytics import YOLO
import cv2
import numpy as np

# load trained model
model = YOLO("runs/segment/train13/weights/best.pt")

cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

captured_frame = None  # This will hold the captured frame
process_captured_frame = False  # This indicates whether to process the captured frame

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame")
        break

    if process_captured_frame:
        img = resize_and_pad(captured_frame, 640, 640)
        # Perform prediction on the captured image
        results = model.predict(img)
        mask_img = results[0].plot(labels=False, boxes=False)

        # Convert the plot image from RGB to BGR format because OpenCV uses BGR
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

        masks = results[0].masks
        if masks is not None:
            num_masks = len(masks)
            print(f"Number of masks: {num_masks}")

            # Set the position and text color
            text_position = (20, 50)  # Top-left corner of the image
            text_content = f" {num_masks}"
            text_color = (0, 0, 255)  # Red color for the text in BGR format
            font_scale = 1  # Depending on your image size, you may need to adjust this
            thickness = 2  # Thickness of the text

            # Put the text on the image
            cv2.putText(mask_img, text_content, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                        thickness)

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
        cv2.imshow("Captured Image with Masks", mask_img)
        process_captured_frame = False

    else:
        # Show the live video feed while waiting for capture
        cv2.imshow("Live Video", frame)

    # Capture an image if 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        captured_frame = frame.copy()
        process_captured_frame = True  # Set the flag to process the captured frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
