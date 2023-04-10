from ultralytics import YOLO
import ultralytics
from PIL import Image
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
    print(result.boxes.xyxy)'
    xmin, ymin, xmax, ymax = bbox[0]
    print(result.orig_img.shape)

