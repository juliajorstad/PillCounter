from ultralytics import YOLO
from ultralytics import SAM
import cv2
from PIL import Image
IMAGE_PATH = "dataset/pill13.jpg"
model = YOLO('yolov8n-seg.pt')

results = model.train(data='dataset/data.yaml', epochs=5, imgsz=640)
