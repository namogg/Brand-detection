from ultralytics import YOLO
from PIL import Image
import cv2
import torch
#E:/Brand detect/Brand-detection/test/videos/mcdonald2.mp4
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
model = YOLO('runs/detect/train13/weights/best.pt')
device = torch.device('cuda')
#result = model.predict(source =  "E:/Brand detect/Brand-detection/test/images/mcdonald.jpg", conf = 0.2, show=True) 
image = Image.open("E:/Brand detect/Brand-detection/test/images/mcdonald.jpg")
# Load image
image = cv2.imread("example.jpg")

# Run inference
results = model(torch.from_numpy(image).unsqueeze(0))

# Apply non-maximum suppression to get the most confident detections
results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

# Extract boxes from the detection results
boxes = results[0].xyxy.numpy().astype(int)