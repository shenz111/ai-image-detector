import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class AIDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root, cls)
            for name in os.listdir(folder):
                self.samples.append(os.path.join(folder, name))
                self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label