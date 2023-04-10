import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import math

def create_model(num_classes = 27, train = True):
    model = models.googlenet(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(model)
    # Set all layers to be trainable
    if train == True:
        for param in model.parameters():
            param.requires_grad = True
    """# Replace last layer with custom classifier for your task

    print(model)
    #
    classifier = list(model.classifier.children())[:-1]

    # Add a new layer at the end for classification
    classifier.extend([nn.Linear(4096, num_classes)])

    # Set the modified classifier as the model's classifier
    model.classifier = nn.Sequential(*classifier)"""
    return model

model = create_model()

#Config
criterion = nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0001,momentum=0.9)
num_epochs = 80
# define the per-pixel normalization transform
def create_transform:
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])]
    )


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

dataset = ImageDataset(img_dir='images/train', label_file='classification label.txt',transform=transform, train=True)
# specify the desired lengths of the two smaller datasets
train_length = int(len(dataset)*0.9)
test_length = len(dataset) - train_length

train_data, test_data = torch.utils.data.random_split(dataset,[train_length,test_length])
dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

class_names = ['Ferrari', 'Ford', 'Nbc', 'Starbucks', 'RedBull', 'Mini', 'Unicef', 'Yahoo', 'Sprite', 'Texaco', 'Intel', 'Cocacola', 'Citroen', 'Heineken', 'Apple', 'Google', 'Fedex', 'Pepsi', 'Puma', 'DHL', 'Porsche', 'Nike', 'Vodafone', 'BMW', 'McDonalds', 'HP', 'Adidas']
def print_images_with_labels(dataloader, class_names):
    # Get a random selection of images from the dataset
    data_iter = iter(dataloader)
    images, labels = data_iter.next()

    # Make a grid of images and their labels
    img_grid = torchvision.utils.make_grid(images)
    img_grid = np.transpose(img_grid, (1, 2, 0))
    plt.imshow(img_grid)

    # Print the labels below each image
    label_grid = [class_names[label] for label in labels]
    plt.xticks([])
    plt.yticks([])
    plt.title('Images with Labels')
    plt.xlabel('\n'.join(label_grid), fontsize=12)
    plt.show()

print_images_with_labels(testloader,class_names)

def show_predictions(model, testloader, class_names):
    # Set the model to evaluation mode
    model.eval()

    # Get a batch of test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    batch_size = images.shape[0]

    # Make predictions on the batch of images
    with torch.no_grad():
        # Convert the input tensor to CUDA tensor
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Create a grid of images with their corresponding predicted labels
    fig, ax = plt.subplots(nrows=2, ncols=batch_size//2, figsize=(15, 8))
    fig.suptitle('Predictions')
    plt.subplots_adjust(hspace=0.4)  # adjust vertical spacing between subplots
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        label = class_names[labels[i].item()]
        prediction = class_names[predicted[i].item()]
        row = i // (batch_size//2)
        col = i % (batch_size//2)
        ax[row][col].imshow(image)
        ax[row][col].set_title(f"{label}\n{prediction}")
        ax[row][col].axis('off')

    plt.show()


best_accuracy = 0.0
for epoch in range(num_epochs):
    print("Running epoch:", epoch + 1)
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_accuracy = 100 * correct / total
    print("Epoch:", epoch + 1, "|| Training Loss:", running_loss, "|| Training Accuracy:", epoch_accuracy, "%")
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss / len(testloader), accuracy))
        #show_predictions(model, testloader, class_names)
        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            print("=========================model is saved=========================")
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_classification_model.pt')
            

torch.save(model.state_dict(), 'classification_model.pt')
print("best accuracy:"+str(best_accuracy))
print("last accuracy:"+str(accuracy))
