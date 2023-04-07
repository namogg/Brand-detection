import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/video
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/images/mcdonald3.img
model = YOLO('runs/detect/train13/weights/best.pt')
img1 = Image.open("bus.jpg")
results = model.predict(source=0,conf = 0.2, show=True) 
cv2.waitKey()

# Extract detected objects
for i, (bbox, class_id, score) in enumerate(zip(results.xyxy[i], results.names[i], results.scores[i])):
    x1, y1, x2, y2 = bbox
    object_img = img.crop((x1, y1, x2, y2))
    object_img.save(f"object_{i}.jpg")
