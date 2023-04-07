import ultralytics
from ultralytics import YOLO


model = YOLO('runs/detect/train13/weights/best.pt')
model.export(format = "saved_model")