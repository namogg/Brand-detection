import ultralytics
from ultralytics import YOLO
import  ultralytics.yolo.data as Dataset
import torch
import comet_ml
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Define transforms for your data
transform = transforms.Compose([
    # Add your transforms here
])

# Create your dataset
dataset = Dataset.LoadImagesAndLabels(data='data.yaml', augment=transform)

# Define the split ratio
train_ratio = 0.8
val_ratio = 1 - train_ratio

# Split your dataset into training and validation sets
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define your samplers
train_sampler = SubsetRandomSampler(range(train_size))
val_sampler = SubsetRandomSampler(range(train_size, dataset_size))

# Create your data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, sampler=val_sampler, num_workers=1)


model = YOLO('runs/detect/train13/weights/best.pt') 

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )
    experiment.set_model_graph(model)
    #cfg = "config.yaml"
    model.train(data='data.yaml', epochs = 150, batch=16, imgsz = 640,  optimizer="SGD", workers = 1)
