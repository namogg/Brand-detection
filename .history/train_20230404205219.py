import ultralytics
from ultralytics import YOLO
import  ultralytics.yolo.data as Dataset
import torch
import comet_ml
import torchvision.transforms as transforms
import albumentations as A

train_transforms = A.Compose([
    A.Resize(height=640, width=640),
    A.Rotate(limit=10, border_mode=0, p=0.5),
    A.HorizontalFlip(p=0.5),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(height=640, width=640),
    ToTensorV2()
])
model = YOLO('runs/detect/train13/weights/best.pt') 
Dataset.YOLODataset
if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment.set_model_graph(model)
    model.train(data='data.yaml', epochs = 150, batch=16, imgsz = 640,  optimizer="SGD", workers = 1)
