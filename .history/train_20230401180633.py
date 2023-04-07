import ultralytics
from ultralytics import YOLO
import torch
import comet_ml

experiment = comet_ml.Experiment(
    api_key="<Your API Key>",
    project_name="<Your Project Name>"
)
model = YOLO('yolov8n.pt') 
if __name__ == '__main__':
    model.train(data = "data.yaml", epochs = 50, batch=4, imgsz = 300,  optimizer="SGD")
