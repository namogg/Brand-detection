import ultralytics  
from ultralytics import YOLO
import PIL  
model = YOLO(".\train\weights\best.pt")
model 