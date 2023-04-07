import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import comet_ml
resnet = models.resnet101(pretrained=True)

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
num_epochs = 100

transform = transforms.Compose([    transforms.Resize((224, 224)),   
                                 transforms.ToTensor()])
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_file, transform=None,split_ratio=0.8):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        
        if self.train:
            # Use a fraction of the data for training
            self.labels = self.labels[:int(len(self.labels)*self.split_ratio)]
        else:
            # Use the remaining fraction of the data for testing
            self.labels = self.labels[int(len(self.labels)*self.split_ratio):]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
device = torch.device('cuda')
resnet.to(device)
experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )

dataset = ImageDataset(img_dir='images/train', label_file='classification_label',transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for epoch in range(num_epochs):
    print("Running")
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print("Running epoch:"+str(epoch)+"|| Training Loss:" +str(running_loss))
    with torch.no_grad():
    test_loss = 0.0
    correct = 0
    total = 0
    for data in test_dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss / len(test_dataloader), accuracy))



torch.save(resnet.state_dict(), 'classification_model.pt')