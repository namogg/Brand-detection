import ultralytics
from ultralytics import YOLO
import torch
import comet_ml

model = YOLO('yolov8n.pt') 
if __name__ == '__main__':
    model.train(data = "data.yaml", epochs = 50, batch=8, imgsz = 640,  optimizer="SGD")
