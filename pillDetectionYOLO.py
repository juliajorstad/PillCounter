from ultralytics import YOLO
import cv2
import numpy as np
from functions import resize_and_pad

# load trained model
model = YOLO("runs/segment/train3/weights/best.pt")


# metrics=model.val()
# train model
# model.train(data='datasets/data.yaml', epochs=20, imgsz=640)


def count_classes(cls_ids):
    map_class_names = {0: "Capsules", 1: "Pills"}

    class_counts = {}

    # loop though the tensor object with class ids
    for class_id in cls_ids:
        # convert the tensor value to int
        class_id = int(class_id)

        # map the class id with the corresponding class name
        class_name = map_class_names.get(class_id)

        # if the class name is in the dictionary, increment value of this class by 1
        if class_name in class_counts:
            class_counts[class_name] += 1
        # if class name not in dictionary, add the class name and value 1
        else:
            class_counts[class_name] = 1

    return class_counts


def show_results(masks, class_counts, img):
    if masks is not None:

        # Set the position and text color
        text_position = (20, 50)  # Top-left corner of the image
        for class_name, count in class_counts.items():
            text_content = f"{class_name}: {count}"
            cv2.putText(img, text_content, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            text_position = (text_position[0], text_position[1] + 30)

        for i in range(len(masks)):
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
            cv2.circle(img, (cX, cY), point_size, (0, 0, 255), -1)  # -1 fills the circle

        # Show the result
        """cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
    return img


def predict_with_yolo(captured_img):
    img = resize_and_pad(captured_img, 640, 640)

    # Make a prediction
    results = model.predict(img)
    masks = results[0].masks

    # returns a tensor object a list of with class value from boxes
    class_ids = results[0].boxes.cls

    # plot the result image, returns numpy array of the image. Set boxes=True to show bounding boxes, class labels and probabilities
    annotated_img = results[0].plot(labels=True, boxes=False)

    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    class_counts = count_classes(class_ids)
    predicted_image = show_results(masks, class_counts, img)

    return predicted_image
