import os
import torch as nn
import torchvision
import torchvision.transforms as transforms
import torch
# Define the transform for preprocessing the data
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Define the path to your data directory
data_dir = "images/train"

# Define the path to your label file
label_file = "labels.txt"

# Read the labels from the label file into a list
with open(os.path.join(data_dir, label_file), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Define the dataset using the ImageFolder class
train_dataset = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=transform
)

# Set the class names for the dataset
train_dataset.classes = class_names

# Define the dataloader
train_dataloader = nn.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8192, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(model)

