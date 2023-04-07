import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
model = models.resnet101()
# Replace last layer with custom classifier for your task
num_classes = 27
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('my_model.pt'))


img = Image.open("C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/images/mcdonald3.jpg")
transform = transforms.Compose([    transforms.Resize((224, 224)),   
                                 transforms.ToTensor()])
model.