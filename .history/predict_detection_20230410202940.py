from ultralytics import YOLO
import ultralytics
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
from predict_classification import *

#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonald4.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
results = model.predict(source = "E:/Brand detect/Brand-detection/test/videos/mcdonald.mp4" , conf = 0.15, show=False) 
#cv2.waitKey()
model_classification = create_model(27,train = False,load_path='classification_model.pt')
print(model_classification)
brand_list = []
for result in results: 
    bbox = result.boxes.xyxy 
    if bbox.numel():
        xmin, ymin, xmax, ymax = bbox[0].int()
        # Crop image using bounding box coordinates
        cropped_img = result.orig_img[ymin:ymax, xmin:xmax, :]
        img = Image.fromarray(cropped_img)
        img_tensor = transform_image(img,train = False)
        brand = show_prediction(img_tensor,model_classification,class_names,threshold=0.)
        if brand not in brand_list:
            brand_list.append(brand)

print("Brand detected: "+str(len(brand_list)))
print(brand_list)
