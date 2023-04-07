import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
#C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/video
#C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/images/mcdonald3.jpg
model = YOLO('runs/detect/train13/weights/best.pt')

results = model.predict(source = "C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/images/mcdonald3.jpg", conf = 0.2, show=True) 

print(results)

