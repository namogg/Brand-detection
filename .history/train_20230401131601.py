import ultralytics
from ultralytics import YOLO
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('yolov8n.pt') 

model.train(data = "data.yaml", epochs = 1, batch=16, imgsz = 640,  optimizer="Adam")