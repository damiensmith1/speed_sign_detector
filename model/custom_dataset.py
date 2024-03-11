from torch.utils.data import Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, label_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.img_labels = []
        with open(label_path, "r") as f:
            for line in f:
                image_name, label = line.strip().split(",")
                self.img_labels.append((image_name, int(label)))
        self.img_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
