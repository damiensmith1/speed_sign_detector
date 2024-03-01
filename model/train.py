import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# Data augmentation and normalization for training
# Randomly apply transformations with a certain probability
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flip the image with probability of 0.5
        transforms.RandomRotation(
            degrees=20
        ),  # Randomly rotate the image within a 20 degree range
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Randomly change brightness, contrast, and saturation
        transforms.ToTensor(),
        AddGaussianNoise(0.0, 0.1),  # Add random noise
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)
train_dataset = ImageFolder(root="../data/train", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")  # Example model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Adjust according to your number of classes

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
if __name__ == "__main__":
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Finished Training")
    model_path = "model_weights.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
