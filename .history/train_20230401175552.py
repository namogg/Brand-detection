import ultralytics
from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt') 
if __name__ == '__main__':
    model.train(data = "data.yaml", epochs = 5, batch=4, imgsz = 300,  optimizer="SGD")
