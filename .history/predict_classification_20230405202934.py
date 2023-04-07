import ultralytics  
from ultralytics import YOLO
from PIL import Image
import cv2
import torchvision.models as models
import torch 
model = models.resnet101()
model.load_state_dict(torch.load('my_model.pt'))