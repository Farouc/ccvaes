# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CartoonMultiLabelDataset(Dataset):
    def __init__(self, root_dir, target_attributes, transform=None):
        """
        Args:
            root_dir (str): Chemin vers le dossier d'images.
            target_attributes (list): Liste des noms d'attributs à extraire 
                                      Ex: ["hair_color", "face_color"]
            transform (callable, optional): Transformations à appliquer sur l'image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_attributes = target_attributes
        
        # Liste d’images
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        
        self.labels = []
        # On va stocker le nombre max de classes pour chaque attribut ici
        # Dictionnaire temporaire pour vérifier la cohérence
        self.dims_per_attr = {attr: 0 for attr in target_attributes}

        print(f"Chargement des labels pour : {target_attributes} ...")

        for img_file in self.images:
            csv_file = img_file.replace(".png", ".csv")
            csv_path = os.path.join(root_dir, csv_file)
            
            # Lecture du CSV (sans header)
            # col 0: nom attribut, col 1: index de la variante, col 2: nombre total de variantes
            df = pd.read_csv(csv_path, header=None)
            
            current_label_vector = []
            
            for attr in self.target_attributes:
                # Récupération de la ligne correspondant à l'attribut
                row = df[df.iloc[:, 0] == attr]
                
                if not row.empty:
                    class_index = int(row.iloc[0, 1])
                    max_classes = int(row.iloc[0, 2])
                    
                    current_label_vector.append(class_index)
                    
                    # On met à jour le nombre de classes max détectées pour cet attribut
                    # (Normalement c'est constant, mais on prend la valeur du fichier)
                    self.dims_per_attr[attr] = max_classes
                else:
                    raise ValueError(f"Attribut '{attr}' non trouvé dans {csv_file}")

            self.labels.append(current_label_vector)

        # Conversion en Tensor Long (N_images, N_attributes)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Liste ordonnée du nombre de classes pour chaque attribut
        # Ex pour ["hair_color", "face_color"] -> [10, 11]
        self.num_classes_list = [self.dims_per_attr[attr] for attr in target_attributes]
        
        print(f"Dataset chargé. Attributs: {self.dims_per_attr}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Retourne un vecteur de labels. 
        # Ex: tensor([4, 8]) si cheveux=4 et visage=8
        label = self.labels[idx] 

        return img, label