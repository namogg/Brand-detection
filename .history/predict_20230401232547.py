import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO("")
img1 = Image.open("bus.jpg")
results = model.predict(source="0", show=True) 
print(results)