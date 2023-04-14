from PIL import Image
import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import io
from train_classification import create_model,create_transform,class_names

model = create_model(27,train = False,load_path='classification_model.pt')

#E:/Brand detect/Brand-detection/test/images/mcdonaldlogo.jpg
img = Image.open("E:/Brand detect/Brand-detection/test/images/fedex.jpg")

def transform_image(image,train = True):
    my_transforms = create_transform()
    if train == False:
        return my_transforms(image).unsqueeze(0)
    return my_transforms
img = transform_image(img,train = False)

def show_prediction(img,model,class_names,show = False,threshold = 0.5): 
    # Pass the image through the model and get the predictions
    with torch.no_grad():
        outputs = model(img)
    
    probs = torch.nn.functional.softmax(outputs, dim=1)
    # Get the predicted class and its corresponding probability
    pred_prob, predicted = torch.max(probs, dim=1)
    if pred_prob.item() > threshold:
        print("Predicted Class: "+ class_names[predicted]+" Confidence: "+str(pred_prob[0].toin()))
        if show:
            img = np.transpose(img.squeeze().numpy(), (1, 2, 0))
            # Display the image along with the predicted label
            fig, ax = plt.subplots()
            ax.imshow(img)
            # Convert the output to probabilities using softmax

            ax.set_title(f"Predicted Brands: {class_names[predicted]}")
            plt.show()
if __name__ == "__main__":
    show_prediction(img,model,class_names)