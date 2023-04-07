import ultralytics
from ultralytics import YOLO
import torch
import comet_ml
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            self.data = f.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip().split()
        img_path = line[0]
        targets = [list(map(int, x.split(','))) for x in line[1:]]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)

        return img, targets

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )

    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.Resize((416,416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset with data augmentation
    train_dataset = CustomDataset(data_path='data.yaml', transform=transform)

    # Create DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize YOLO model
    model = YOLO('runs/detect/train13/weights/best.pt') 

    # Set the model graph for Comet.ml logging
    experiment.set_model_graph(model)

    # Train the model
    model.train(data=train_loader, epochs=150, imgsz=640, optimizer="SGD", workers=1)

