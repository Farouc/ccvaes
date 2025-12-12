import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class UTKFaceDataset(Dataset):
    """
    Dataset class for UTKFace (Age Regression).
    Parses age from filename and normalizes it.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # List all JPG images
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        print(f"UTKFace Dataset loaded: {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. Load Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 2. Extract Age (Filename format: 26_0_1_....jpg)
        # We take the first number before the first underscore
        try:
            age = int(img_name.split('_')[0])
        except ValueError:
            print(f"Warning: Could not parse age from {img_name}, defaulting to 0")
            age = 0
        
        # 3. Normalization (Crucial for regression!)
        # Divide by 100 to get a value between 0.0 and 1.0
        # Example: 26 years old -> 0.26
        # Cap at 100 years (1.0) to avoid outliers breaking the scale
        age_label = min(age, 100) / 100.0
        
        # Return Float Tensor (required for MSE Loss)
        return image, torch.tensor([age_label], dtype=torch.float32)