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
# Define the path to your label file
label_file = "classification_label"


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

train_dataset = CustomImageDataset(annotations_file = label_file,img_dir = data_dir, transform=transforms)
# Set the class names for the dataset
train_dataset.classes = list(label_map.keys())


# Define the dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(model)

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        y_pred = model(inputs.to(device))
        loss = loss_fn(y_pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))