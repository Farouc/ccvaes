# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # On liste toutes les images JPG
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        print(f"Dataset UTKFace chargé : {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. Chargement Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 2. Extraction Age (Nom du fichier : 26_0_1_....jpg)
        # On prend juste le premier chiffre avant le "_"
        age = int(img_name.split('_')[0])
        
        # 3. Normalisation (Crucial pour la regression !)
        # On divise par 100 pour avoir une valeur entre 0.0 et 1.0
        # Ex: 26 ans -> 0.26
        # On plafonne à 100 ans (1.0) pour éviter les rares centenaires qui casseraient l'échelle
        age_label = min(age, 100) / 100.0
        
        # On retourne un Float Tensor (nécessaire pour la Loss MSE)
        return image, torch.tensor([age_label], dtype=torch.float32)