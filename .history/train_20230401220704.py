import ultralytics
from ultralytics import YOLO
import torch
import comet_ml


model = YOLO('runs/detect/train9/weights/best.pt') 
if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment = 
    model.train(data = "data.yaml", epochs = 100, batch=8, imgsz = 640,  optimizer="SGD", workers = 1)
