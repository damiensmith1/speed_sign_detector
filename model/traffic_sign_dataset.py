
from torch.utils.data import Dataset

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, label_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.labels_file = label_path
        self.image_paths = []
        self.labels = []
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return idx
