from ultralytics import YOLO
import ultralytics
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
results = model.predict(source = "E:/Brand detect/Brand-detection/test/images/mcdonald.jpg" , conf = 0.2, show=True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
cv2.waitKey()
print(len(results))
for result in results: 
    bbox = result.boxes.xyxy 
    xmin, ymin, xmax, ymax = bbox[0]
    # Crop image using bounding box coordinates
    cropped_img = result.orig_img[ymin:ymax, xmin:xmax, :]
    # show the image
    cv2.imshow(cropped_img)


