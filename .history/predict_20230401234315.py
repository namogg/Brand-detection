import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/video
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/images/mcdonald3.img
model = YOLO('runs/detect/train12/weights/best.pt')
img1 = Image.open("bus.jpg")
results = model.predict(source="C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/video/mcdonald.mp4", show=True) 
cv2.waitKey()