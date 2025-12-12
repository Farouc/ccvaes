# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class UTKFaceDataset(Dataset):
    """
    PyTorch Dataset for the UTKFace dataset (age regression).

    Each image filename follows the format:
        age_gender_race_date.jpg

    Example:
        26_0_1_20170116174525125.jpg

    We extract:
        - the image
        - the age label (normalized to [0, 1])
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the UTKFace image directory.
            transform (callable, optional): Image transformations.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all JPEG images in the directory
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith(".jpg")
        ]

        print(f"UTKFace dataset loaded: {len(self.image_files)} images.")

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load one image and its corresponding age label.

        Returns:
            image (Tensor): Transformed image tensor.
            age_label (Tensor): Normalized age in [0, 1], shape (1,).
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # --------------------------------------------------
        # 1. Load image
        # --------------------------------------------------
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # --------------------------------------------------
        # 2. Extract age from filename
        # Filename format: age_gender_race_*.jpg
        # --------------------------------------------------
        age = int(img_name.split("_")[0])

        # --------------------------------------------------
        # 3. Normalize age for regression
        # - Divide by 100 to map age to [0, 1]
        # - Cap at 100 to avoid extreme outliers
        # --------------------------------------------------
        age_normalized = min(age, 100) / 100.0

        # Return age as a float tensor (required for MSE loss)
        return image, torch.tensor([age_normalized], dtype=torch.float32)
