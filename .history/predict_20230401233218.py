import ultralytics  
from ultralytics import YOLO
from PIL import Image
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/video
#C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/images/mcdonald3.img
model = YOLO('runs/detect/train12/weights/best.pt')
img1 = Image.open("bus.jpg")
results = model.predict(source="C:/Users/ADMIN/Documents/TensorFlow\workspace/training_demo/images/mcdonald3.jpg", show=True) 