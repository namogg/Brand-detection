import ultralytics
from ultralytics import YOLO

# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="pk7hCu2mWb4IsliKxNRWHfnCv",
    project_name="Logo regconize",
    workspace="namogg",
)
model = YOLO('yolov8n.pt') 

model.train(data = "data.yaml", epochs = 3, batch=16, imgsz = 640,  optimizer="Adam")