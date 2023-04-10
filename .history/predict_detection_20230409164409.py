from ultralytics import YOLO
from PIL import Image
import cv2
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')

results = model.predict(source =  "E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg", conf = 0.2, show=True) 
cv2.waitKey()

crop = results.crop()