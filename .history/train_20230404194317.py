import ultralytics
from ultralytics import YOLO
import torch
import comet_ml
import torchvision.transforms as transforms
model = YOLO('runs/detect/train13/weights/best.pt') 
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.Resize((416,416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment.set_model_graph(model)
    train_dataset = model.create_dataset(data='data.yaml', transform=transform)
    model.train(data = train_dataset, epochs = 150, batch=16, imgsz = 640,  optimizer="SGD", workers = 1)
