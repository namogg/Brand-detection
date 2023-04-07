import ultralytics  
from ultralytics import YOLO
from PIL import Image

model = YOLO(".\train\weights\best.pt")
img1 = 