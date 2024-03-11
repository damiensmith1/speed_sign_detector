import cv2
import torch
from PIL import Image
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

model_path = "./model/model_weights.pth"
model.load_state_dict(torch.load(model_path))

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="./yolov5/runs/train/exp/weights/best.pt"
)

transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ]
)


yolo_model.eval()
model.eval()

input_video_path = "60-example.mp4"
output_video_path = "60-demo.mp4"

cap = cv2.VideoCapture(input_video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

label_map = {0: 40, 1: 50, 2: 60, 3: 100}


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    results = yolo_model(img_pil)
    detections = results.xyxy[0].cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = map(int, detection[:6])

        cropped_np = frame[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped_np)
        crop_img_tensor = transform(cropped_pil).unsqueeze(0)
        with torch.no_grad():
            classifier_output = model(crop_img_tensor)
            _, predicted = torch.max(classifier_output.data, 1)
            result = predicted.item()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Speed: {label_map[result]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
