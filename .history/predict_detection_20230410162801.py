from ultralytics import YOLO
import ultralytics
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
from predict_classification import *

matplotlib.use('TkAgg')  # or 'Qt5Agg'
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
results = model.predict(source = "E:/Brand detect/Brand-detection/test/images/mcdonald.jpg" , conf = 0.2, show=True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
cv2.waitKey()
model_classification = create_model(27,train = False,load_path='classification_model.pt')

for result in results: 
    bbox = result.boxes.xyxy 
    xmin, ymin, xmax, ymax = bbox[0].int()
    # Crop image using bounding box coordinates
    cropped_img = result.orig_img[ymin:ymax, xmin:xmax, :]
    tensor = torch.from_numpy(cropped_img)
    img = 
    print(cropped_img)
    show_prediction(tensor,model_classification,class_names)

