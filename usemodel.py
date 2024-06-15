import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Define the same model architecture as used for training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 14 * 14)  # Adjust to match the flattened size
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.load_state_dict(torch.load('C:\\Users\\humbl\\Documents\\Pytorch Testing\\Brain MRI\\BrainMRI.pt'))
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Run the image through the model
def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction

# Example usage
image_path = 'C:\\Users\\humbl\\Downloads\\9b3c24072a2568610b99707aa719ce850ac5b3ac515d6abb3692649109cf19e9_big_gallery.jpeg'
prediction = predict(image_path)
disease = ''
if prediction == 0:
    disease = 'glioma'
elif prediction == 1:
    disease = 'meningioma'
elif prediction == 2:
    disease = 'no tumor'
else:
    disease = 'pituitary'
print(f'Prediction: {disease}')

# Display the image
image = Image.open(image_path)
plt.imshow(image, cmap='gray')
plt.title(f'Prediction: {disease}')
plt.show()
