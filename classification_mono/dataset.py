# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CartoonHairColorDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_attribute = "hair_color"

        # List of images
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        self.labels = []

        for img_file in self.images:
            csv_file = img_file.replace(".png", ".csv")
            df = pd.read_csv(os.path.join(root_dir, csv_file), header=None)

            # hair_color
            row = df[df.iloc[:, 0] == self.target_attribute]
            class_index = int(row.iloc[0, 1])
            max_classes = int(row.iloc[0, 2])

            self.labels.append(class_index)

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.num_classes = max_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label
