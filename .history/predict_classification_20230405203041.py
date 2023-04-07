import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
import torchvision.models as models
import torch 
model = models.resnet101()
# Replace last layer with custom classifier for your task
num_classes = 27
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

model.load_state_dict(torch.load('my_model.pt'))