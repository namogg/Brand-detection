from PIL import Image
import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
model = models.vgg16()

# Replace last layer with custom classifier for your task
num_classes = 27
print(model)
#model.fc = nn.Linear(model.fc.in_features, num_classes)
classifier = list(model.classifier.children())[:-1]

# Add a new layer at the end for classification
classifier.extend([nn.Linear(4096, num_classes)])

# Set the modified classifier as the model's classifier
model.classifier = nn.Sequential(*classifier)
# Replace last layer with custom classifier for your task   
#model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('classification_model.pt'))

#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
img = Image.open("E:/Brand detect/Brand-detection/test/images/apple_fintech.jpg")
normalize = transforms.Lambda(lambda x: x / x.max())
transform = transforms.Compose([ transforms.Resize((224, 224)),   
                                 transforms.ToTensor(),
                                 normalize])
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

# Pass the image through the model and get the predictions
with torch.no_grad():
    outputs = model(img)
    print(outputs)

# Print the top 5 predictions
_, pred = outputs.topk(5, 1, True, True)
print(pred)