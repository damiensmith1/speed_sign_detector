import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np


# Data augmentation and normalization for training
# Randomly apply transformations with a certain probability
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)
train_dataset = ImageFolder(root="../data/test", transform=train_transforms)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")  # Example model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Adjust according to your number of classes

model_path = "model_weights.pth"
# Load the saved weights
model.load_state_dict(torch.load(model_path))

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def calculate_accuracy(model, data_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Evaluate the model after training
accuracy = calculate_accuracy(model, test_loader)
print(f"Accuracy on test set: {accuracy:.2f}%")
