import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm  # Pour voir la barre de progression au chargement

class CartoonDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_images=None):
        """
        root_dir: Le dossier contenant les .png et .csv mélangés
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. Lister tous les fichiers PNG
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        
        if num_images:
            all_files = all_files[:num_images]
            
        self.files = all_files
        self.labels = []

        print(f"Chargement et Normalisation des labels pour {len(self.files)} images...")
        
        # 2. Pré-charger tous les CSVs en mémoire
        for img_file in tqdm(self.files):
            # Le csv a le même nom que l'image, juste l'extension change
            csv_file = img_file.replace('.png', '.csv')
            csv_path = os.path.join(root_dir, csv_file)
            
            try:
                # Le CSV contient 3 colonnes : "Attribut", "Index Variante", "Nombre total variantes"
                # Pas de header dans les fichiers Cartoon Set
                df = pd.read_csv(csv_path, header=None)
                
                # Colonne 1 : La valeur actuelle de l'attribut (ex: index 4 pour cheveux blonds)
                vals = df.iloc[:, 1].values.astype(float)
                
                # Colonne 2 : Le nombre total de variantes possibles pour cet attribut (ex: 10 couleurs possibles)
                max_vals = df.iloc[:, 2].values.astype(float)
                
                # --- NORMALISATION CRITIQUE [0, 1] ---
                # On divise par (max - 1) pour ramener l'index entre 0.0 et 1.0.
                # Exemple : si j'ai l'option 4 sur 5 (indices 0,1,2,3,4), 4/(5-1) = 1.0
                # On ajoute 1e-6 pour la sécurité (éviter division par zéro)
                normalized_vals = vals / (max_vals - 1 + 1e-6)
                
                self.labels.append(torch.tensor(normalized_vals, dtype=torch.float32))
                
            except Exception as e:
                print(f"Erreur lecture {csv_file}: {e}")
                # Si erreur, on met des zéros (fallback) pour ne pas planter
                self.labels.append(torch.zeros(18)) 

        # Convertir la liste en un gros tenseur pour la vitesse d'entraînement
        self.labels = torch.stack(self.labels)
        print(f"Labels chargés ! Dimensions : {self.labels.shape}") # (10000, 18 généralement)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Charger l'image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Récupérer le label pré-chargé et normalisé
        attributes = self.labels[idx]

        return image, attributes