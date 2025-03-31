import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Path to the model and image
model_path = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/PRODUCT_IDENTIFICATION/resnet18_model.pth'
image_path = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/PRODUCT_IDENTIFICATION/test.png'
dataset_directory = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/DATA/DATASET_UNANNOTATED/train'

# Extract class names from directory names
class_names = sorted([d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))])

# Load the ResNet-18 model
model = models.resnet18()

# Modify the last fully connected layer to match your custom model
num_classes = len(class_names)  # Update num_classes based on the number of directories
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the state dict with the correct architecture
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(image)

# Get the class with the highest probability
probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Convert logits to probabilities
max_prob, predicted_class = torch.max(probabilities, 0)

# Print the highest probability class and its name
print(f"Predicted Class: {class_names[predicted_class.item()]}, Probability: {max_prob.item():.4f}")
