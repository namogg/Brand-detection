import ultralytics
from ultralytics import YOLO
import torch
import comet_ml
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
model = YOLO('runs/detect/train13/weights/best.pt') 

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.RandomRotation(degrees=20),
    transforms.RandomCrop(size=(384, 384)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Resize((416,416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ultralytics.

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment.set_model_graph(model)
    model.transforms = 
    model.train(data='data.yaml', epochs = 150, batch=16, imgsz = 640,  optimizer="SGD", workers = 1)
