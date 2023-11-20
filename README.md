# PillPal 
This project was developed as a project for educational purposes.
PillPalApp is a conceptual mobile application to identify pills using a camera or uploaded images. 
It employs YOLO (You Only Look Once) model for object detection and segmentation, integrated into a user-friendly interface built with 
Kivy and KivyMD as a desktop application for visualization of the project.   

Key Features:
* Camera Capture: Users can capture images using their device's camera.
* Image Upload: Users have the option to upload images from their device.
* YOLO Model Integration: Utilizes the YOLO model for real-time object detection.

## Requirements
* Python > 3.8
* Install ultralytics: https://pypi.org/project/ultralytics/
* Install OpenCv
* Numpy
* Kivy: https://pypi.org/project/Kivy/
* KivyMD: https://pypi.org/project/kivymd/

## Set up guide
1. Clone the repository and install the required packages
2. Run "pillpalApp.py" to run the desktop GUI

## Documentation
### Dataset
The images in the dataset is a combination of our own data and datasets found on Roboflow platform. 
Link to the datasets can be found here:
https://universe.roboflow.com/national-chung-cheng-university-miweg/color_medicine
https://universe.roboflow.com/apple-developer-academy-postech-otfp4/pill-instance-segmentation

The datasets are annotated for segmentation in YOLOv8 format using Roboflow, and exported locally for training the model. 
Link to our dataset with labels can be downloaded here: https://universe.roboflow.com/ikt213/ikt2313project/dataset/5

The data is splitted into 80/10/10 train,test and val respectivly. 

### YOLO model
In this project, we used a pre-trained segmentation model "yolov8s-seg.pt" as initial weights, and trained further with our own dataset.
The module "pillDetectionYOLO.py" is where the model is trained and deployed. 
The best weights from the current model is located in 'runs/segment/train3/weights/best.pt'. Change this path in "pillDetectionYOLO.py" after training a new model.

The functions in "pillDetectionYOLO.py" takes in the captured image either from webcamera or uploaded image from the app "pillpalApp.py". The image is then resized, passed to the trained model for predictions, and calculations is performed on the inference image to find the centroid of the mask and number of classes detected.
The original image is then presented in the app with number of objects detected for the two classes, pills and capsules, and a marker is placed in the center of each detected object. 

### Kivy desktop app
We used Kivy and KivyMD to visualize the results in a simple desktop app. The code was written using ChatGPT and Kivy Documentations (link:https://kivymd.readthedocs.io/en/1.1.1/getting-started/) as this was not the main focus of the project. 
The app presents the user with two options: capture the image using the built in camera of the device with a retake button that enables the user to take a new image, and upload button to upload an image from the folder. The current path to uploading images is set to the test dataset that is added in the project. 

## Limitations
* As this project was just a concept for a mobile app, the application is only built with a desktop interface
* The dataset needs refinement as the current trained model struggles to classify correctly. 



