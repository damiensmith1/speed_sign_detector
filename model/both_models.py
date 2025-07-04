import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

model_path = "model_weights.pth"
model.load_state_dict(torch.load(model_path))

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="../yolov5/runs/train/exp/weights/best.pt"
)

yolo_model.eval()
model.eval()
test_img = Image.open("../")
results = yolo_model(test_img)
print(results)

detections = results.xyxy[0]

if len(detections) > 0:
    highest_confidence = max(detections, key=lambda detection: detection[4])
    xyxy = highest_confidence[:4]

    box = [int(coordinate.item()) for coordinate in xyxy]

    crop_img = test_img.crop((box[0], box[1], box[2], box[3]))

    crop_img.save("cropped_image.jpg")

    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ]
    )

    crop_img_tensor = transform(crop_img).unsqueeze(0)
    with torch.no_grad():
        classifier_output = model(crop_img_tensor)
        _, predicted = torch.max(classifier_output.data, 1)
        result = predicted.item()
        print(result)
else:
    print("No detections!")
