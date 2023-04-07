import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as model

resnet = model.resnet101(pretrained=True)

# Replace last layer with custom classifier for your task
num_classes = 27
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Set all layers to be trainable
for param in resnet.parameters():
    param.requires_grad = True
#Config
criterion = nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

transform = transforms.Compose([    transforms.Resize((224, 224)),   
                                 transforms.ToTensor()])
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
dataset = ImageDataset(img_dir='images/train', label_file='classification_label')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0