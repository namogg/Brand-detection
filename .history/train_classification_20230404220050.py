import torch
import torch.nn as nn
import torch.optim as optim
import comet_ml
import torchvision
import torchvision.transforms as transforms
import ultralytics

# Define the ResNet-based model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = ultralytics.models.CSPNet(num_layers=50, num_classes=27, pretrained=True)
        self.linear = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    experiment = comet_ml.Experiment(
        api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
        project_name="logo-regconize"
    )

    # Define the dataset and data loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(root="images/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Create an instance of the ResNet-based model
    model = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
