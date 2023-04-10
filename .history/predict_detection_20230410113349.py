from ultralytics import YOLO
from PIL import Image
import cv2
import torch
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
result = model.predict(source = "E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4" , conf = 0.2, show=True,save_txt = True, save_crop = True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
cv2.waitKey()
# Get the first detection from the result
detection = result.pred[0]

# Get the bounding box coordinates
x1, y1, x2, y2 = detection[:4]


