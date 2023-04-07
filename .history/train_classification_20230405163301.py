import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd     
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
# Define the transform for preprocessing the data
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Define the path to your data directory
data_dir = "images/train"

label_map = {
    'Ferrari': 0,
    'Ford': 1,
    'Nbc': 2,
    'Starbucks': 3,
    'RedBull': 4,
    'Mini': 5,
    'Unicef': 6,
    'Yahoo': 7,
    'Sprite': 8,
    'Texaco': 9,
    'Intel': 10,
    'Cocacola': 11,
    'Citroen': 12,
    'Heineken': 13,
    'Apple': 14,
    'Google': 15,
    'Fedex': 16,
    'Pepsi': 17,
    'Puma': 18,
    'DHL': 19,
    'Porsche': 20,
    'Nike': 21,
    'Vodafone': 22,
    'BMW': 23,
    'McDonalds': 24,
    'HP': 25,
    'Adidas': 26
}
def get_id_by_brand(name, name_map):
    return name_map[name]
# Define the path to your label file
label_file = "classification label.txt"

# Read the labels from the label file into a list
with open(os.path.join(label_file), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_dataset = CustomImageDataset(annonation = "classification label.txt")
# Set the class names for the dataset
train_dataset.classes = class_names

# Define the dataloader
train_dataloader = torch.utils.data.DataLoader(
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(model)

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        y_pred = model(inputs.to(device))
        loss = loss_fn(y_pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            y_pred = model(inputs.to(device))
            acc += (torch.argmax(y_pred, 1) == labels.to(device)).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))