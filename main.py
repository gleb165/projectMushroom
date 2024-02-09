from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import io
import torch.nn as nn
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

app = FastAPI()


def plot_image_with_text(image, predicted_class_name):
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.axis('off')
    plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
    plt.savefig('/Users/glebsavelev/PycharmProjects/fastApiProject2/image.png')
    plt.close()


class ToRGB(object):
    def __call__(self, pic):
        if pic.mode != 'RGB':
            pic = pic.convert('RGB')
        return pic


@app.post("/process_image/")
# import photo
async def process_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))


    # Load the saved model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
    model.load_state_dict(torch.load('model2good'))
    model.eval()

    # Create a new model with the correct final layer
    new_model = models.resnet18(pretrained=True)
    new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

    # Copy the weights and biases from the loaded model to the new model
    new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
    new_model.fc.bias.data = model.fc.bias.data[0:2]
    # Load and preprocess the unseen image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToRGB(),  # Custom transform to convert PNG images to RGB
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Map the predicted class to the class name
    class_names = ['edible', 'poison']  # Make sure these class names match your training data
    predicted_class_name = class_names[predicted_class.item()]
    # Display the image with the predicted class name
    plot_image_with_text(pil_image, predicted_class_name)

    return FileResponse('/Users/glebsavelev/PycharmProjects/fastApiProject2/image.png', media_type="image/jpeg")
