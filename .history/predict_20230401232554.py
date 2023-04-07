import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO("'runs/detect/train12/weights/best.pt'")
img1 = Image.open("bus.jpg")
results = model.predict(source="0", show=True) 
print(results)