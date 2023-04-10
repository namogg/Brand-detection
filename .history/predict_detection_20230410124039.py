from ultralytics import YOLO
from PIL import Image
import cv2
import torch
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
result = model.predict(source = "E:/Brand detect/Brand-detection/test/images/mcdonald.jpg" , conf = 0.2, show=True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
cv2.waitKey()
print(type(result))
result_dict = result[0][1]
print(result_dict)

