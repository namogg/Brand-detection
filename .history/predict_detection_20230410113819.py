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

print(result)


# loop through each detected object
for box in result['boxes']:
    # get the coordinates of the bounding box
    x1, y1, x2, y2 = box.int().tolist()

    # crop the image using the bounding box coordinates
    cropped_image = image[y1:y2, x1:x2]
    
    # save the cropped image
    cv2.imwrite('cropped_image.jpg', cropped_image)