from ultralytics import YOLO
import cv2
import numpy as np
from roi import get_roi, resize_and_pad

# load trained model
model = YOLO("../runs/segment/train13/weights/best.pt")

# train model
#model.train(data='datasets/data.yaml', epochs=3, imgsz=640)


def get_class(cls_ids):
    class_names = {0: "Capsules", 1: "Pills"}

    class_counts = {}

    for cls_id in cls_ids:

        cls_id = int(cls_id)
        class_name = class_names.get(cls_id, f"Unknown class {cls_id}")
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    return class_counts
def show_results(masks,class_counts,img):
    if masks is not None:

        #class_counts = get_class()

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
    cls_ids = results[0].boxes.cls

    mask_img = results[0].plot(labels=True, boxes=False)

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

    class_counts = get_class(cls_ids)
    predicted_image = show_results(masks,class_counts,img)

    return predicted_image




