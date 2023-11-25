# PillPal  

This project was developed as a part of the course Machine Vision. 

PillPalApp is a conceptual mobile application to identify and count pills or capsules using a camera or uploaded images. 
It employs YOLO (You Only Look Once) model for object detection and segmentation, integrated into a user-friendly interface built with 
Kivy and KivyMD as a desktop application for visualization of the project.   


Key Features:
* Camera Capture: Users can capture images using their device's camera.
* Image Upload: Users have the option to upload images from their device.
* YOLO Model Integration: Utilizes the YOLO model for real-time object detection.

Team members: 

* Julia Jørstad
* Tuva Aarseth
* Tønnes Abrahamsen

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
The images in the dataset is a combination of our own data and datasets from Roboflow platform. All the images can be found in the images folder.
Link to the datasets can be found here:
https://universe.roboflow.com/national-chung-cheng-university-miweg/color_medicine
https://universe.roboflow.com/apple-developer-academy-postech-otfp4/pill-instance-segmentation

The datasets are annotated for segmentation in YOLOv8 format using Roboflow, and exported locally for training the model. The data is splitted into 80/10/10 train,test and validation respectivly. The splitted datasets can be found in the project folder "datasets".
Link to our dataset with labels can be downloaded here: https://universe.roboflow.com/ikt213/ikt2313project/dataset/5

```datasets/data.yaml``` were created with the path to the train,test and validation images, number of classes defined as "nc" and the labels.

The images in the folder ```/testImages``` were not used in the model and does not contain any annotations. They were only used for testing the model within the app.
The images in this folder are our own. 
 

### YOLO model
In this project, we used a pre-trained segmentation model "yolov8s-seg.pt" as initial weights from Ultralytics, and trained further with our own dataset.
The module ```pillDetectionYOLO.py``` is where the model is trained and deployed.  

The best weights from the current model trained with 10 epochs is located in ```runs/segment/train3/weights/best.pt```.   
The results from the model can be viewed in ```runs/segment/train3```, and the predicted images from validation can be viewed in ```runs/segment/val``` folder.
The results from training for 20 epochs which caused overfitting is located in train4 folder.


### Code
The functions in ```pillDetectionYOLO.py``` takes in the captured image either from webcamera or uploaded image from the app ```pillpalApp.py```. The image is then resized using ```roi.py```, passed to the trained model for predictions, and calculations is performed on the inference image to find the centroid of the mask and number of classes detected using functions ```count_classes``` and ```show_results```.
The original image is then presented in the app with number of objects detected for the two classes, pills and capsules, and a marker is placed in the center of each detected object from function ```show_results``` in ```pillDetectionYOLO.py```. 

### Kivy desktop app
We used Kivy and KivyMD to visualize the results in a simple desktop app. The code was written using ChatGPT and Kivy Documentations (link:https://kivymd.readthedocs.io/en/1.1.1/getting-started/) as this was out of the scope for this project.  

The app presents the user with two options: capture the image using the built in camera of the device with a retake button that enables the user to take a new image, and upload button to upload an image from the folder. The current path to uploading images is set to the test dataset that is added in the project. 


 <img width="495" alt="Skjermbilde_2023-11-20_kl _15 24 20" src="https://github.com/juliajorstad/PillCounter/assets/58601228/ee090eef-1001-4fcc-804d-ded47f950365">


The logo of the app was designed by Julia Jørstad

## Limitations
* As this project was just a concept for a mobile app, the application is only built with a desktop interface
* The dataset needs refinement: the classes is imbalanced,missed labels in some objects, needs more data in general
* The model is not working perfectly because of issues with the dataset
* Lacking background images: current model detects and counts every object in the image as pills or capsules. Background images need to be added for the model to ignore.




