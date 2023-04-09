from PIL import Image
import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from train_classification import create_model

model = models.resnet34(pretrained=True)
num_classes = 27
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('classification_model.pt'))

#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
img = Image.open("E:/Brand detect/Brand-detection/test/images/fedex.jpg")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(100,100),
    transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

# Pass the image through the model and get the predictions
with torch.no_grad():
    outputs = model(img)
    print(outputs)

# Print the top 5 predictions
_, pred = outputs.topk(5, 1, True, True)
print(pred)