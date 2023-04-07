import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO("E:\YOLO ultralytic\runs/detect/train/weights/best.pt")
img1 = Image.open("bus.jpg")
result = model.predict(source=img1)
result.show()