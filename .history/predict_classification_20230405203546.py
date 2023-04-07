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

#C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/images/mcdonaldlogo.jpg
img = Image.open("C:/Users/ADMIN/Documents/TensorFlow/workspace/training_demo/images/mcdonaldlogo.jpg")
transform = transforms.Compose([    transforms.Resize((224, 224)),   
                                 transforms.ToTensor()])
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

# Pass the image through the model and get the predictions
with torch.no_grad():
    outputs = model(img)

# Print the top 5 predictions
_, pred = outputs.topk(5, 1, True, True)
print(pred)