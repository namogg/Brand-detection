from ultralytics import YOLO
from PIL import Image
import cv2
import torch
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
result = model.predict(source =  "E:/Brand detect/Brand-detection/test/images/mcdonald.jpg", conf = 0.2, show=True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
cv2.waitKey()
boxes = result.boxes.xyxy.to.numpy().astype(int)
confidences = result.boxes.conf.to('cpu').numpy().astype(float)
labels = result.boxes.cls.to('cpu').numpy().astype(int) 

for box, conf, label in zip(boxes, confidences, labels):
    x_min, y_min, x_max, y_max = box
    image_crop = image[y_min:y_max, x_min:x_max]
