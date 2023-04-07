import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO("train/weights/best.pt")
img1 = Image.open("bus.jpg")
results = model.predict(source=img1, save=True) 