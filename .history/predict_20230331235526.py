import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO("E:/YOLO ultralytic/runs/detect/train/weights/best.pt")
img1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True) 
result.show()