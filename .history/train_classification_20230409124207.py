import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models



model = models.resnet34(pretrained=True)

# Replace last layer with custom classifier for your task
num_classes = 27

model.fc = nn.Linear(model.fc.in_features, num_classes)
"""classifier = list(model.classifier.children())[:-1]

# Add a new layer at the end for classification
classifier.extend([nn.Linear(4096, num_classes)])

# Set the modified classifier as the model's classifier
model.classifier = nn.Sequential(*classifier)
# Set all layers to be trainable
for param in model.parameters():
    param.requires_grad = True"""
#Config
criterion = nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9 ,weight_decay=0.0001)
num_epochs = 150
# define the per-pixel normalization transform
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_file, transform=None,train = True):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.transform = transform
        self.train = train

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
model.to(device)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
transform = transforms.Compose([ transforms.Resize((128, 128)),  
                                transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])
dataset = ImageDataset(img_dir='images/train', label_file='classification label.txt',transform=transform, train=True)


# specify the desired lengths of the two smaller datasets
train_length = int(len(dataset)*0.9)
test_length = len(dataset) - train_length

train_data, test_data = torch.utils.data.random_split(dataset,[train_length,test_length])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)

for epoch in range(num_epochs):
    print("Running")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss / len(testloader), accuracy))



torch.save(model.state_dict(), 'classification_model.pt')