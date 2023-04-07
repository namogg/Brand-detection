import ultralytics
from ultralytics import YOLO
import  ultralytics.yolo.data
import torch
import comet_ml
import torchvision.transforms as transforms
import albumentations as A


model = YOLO('runs/detect/train13/weights/best.pt') 

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment.set_model_graph(model)
    model.train(data='data.yaml', epochs = 150, batch=16, imgsz = 640,  optimizer="SGD", workers = 1)
