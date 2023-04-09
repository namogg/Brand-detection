from PIL import Image
import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

model = models.googlenet(pretrained=True)
num_classes = 27
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('best_classification_model.pt'))

class_names = ['Ferrari', 'Ford', 'Nbc', 'Starbucks', 'RedBull', 'Mini', 'Unicef', 'Yahoo', 'Sprite', 'Texaco', 'Intel', 'Cocacola', 'Citroen', 'Heineken', 'Apple', 'Google', 'Fedex', 'Pepsi', 'Puma', 'DHL', 'Porsche', 'Nike', 'Vodafone', 'BMW', 'McDonalds', 'HP', 'Adidas']
#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
img = Image.open("E:/Brand detect/Brand-detection/test/images/apple_fintech.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

inputs, classes = next(iter(val_loader))
 
model = model.to(device)
inputs=inputs.to(device)
 
outputs=model(inputs)
_, preds = torch.max(outputs, 1)
preds=preds.cpu().numpy()
classes=classes.numpy()
print(preds)
print(classes)
# Pass the image through the model and get the predictions
with torch.no_grad():
    outputs = model(img)
    print(outputs)

# Print the top 5 predictions
_, predicted = torch.max(outputs, 1)
print(predicted)

img = np.transpose(img.squeeze().numpy(), (1, 2, 0))

# Display the image along with the predicted label
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title(f"Predicted Class: {class_names[predicted]}")
plt.show()