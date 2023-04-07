import ultralytics
from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt') 
if __name__ == '__main__':
    model.train(data = "data.yaml", epochs = 3, batch=16, imgsz = 640,  optimizer="Adam")
