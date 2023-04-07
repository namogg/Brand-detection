import ultralytics
from ultralytics import YOLO
import torch
import comet_ml

model = YOLO('runs.pt') 
if __name__ == '__main__':
    model.train(data = "data.yaml", epochs = 50, batch=8, imgsz = 640,  optimizer="SGD", workers = 1)
