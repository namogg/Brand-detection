import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/video
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/images/mcdonald3.img
model = YOLO('runs/detect/train13/weights/best.pt')

results = model.predict(source="C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/image/mcdonald3.jpg",conf = 0.2, show=True) 
cv2.waitKey()
