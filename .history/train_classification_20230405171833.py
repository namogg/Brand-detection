import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as model

resnet = models.resnet50(pretrained=False)

# Replace last layer with custom classifier for your task
num_classes = 27
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Set all layers to be trainable
for param in resnet.parameters():
    param.requires_grad = True
#