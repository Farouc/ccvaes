import torch
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Imports de ton modèle
from model import CCVAE

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./cartoonset10k/cartoonset10k/cartoonset10k"  # Ton dossier d'images
MODEL_PATH = "/users/eleves-a/2022/rida.assalouh/ccvaes/ccvae_haircolor_new.pth"           # Ton modèle entraîné avec Contrastive Loss
SAMPLE_LIMIT = 20000                          # Nombre d'images pour le test

# Les attributs qu'on veut tester
PROBE_ATTRIBUTES = ["hair_color", "glasses", "face_shape", "face_color"]

print(f"--- Starting Leakage Audit on {DEVICE} ---")

# --- 2. DÉFINITION D'UN DATASET FLEXIBLE (Juste pour ce test) ---
class CartoonProbeDataset(Dataset):
    def __init__(self, root_dir, target_attributes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_attributes = target_attributes
       
        # On liste les images
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        self.labels_dict = {attr: [] for attr in target_attributes}
       
        print(f"Lazy loading labels for {target_attributes}...")
       
        # On lit les CSVs
        # Note: Pour aller vite, on pourrait optimiser, mais c'est simple et sûr.
        count = 0
        for img_file in self.images:
            csv_file = img_file.replace(".png", ".csv")
            csv_path = os.path.join(root_dir, csv_file)
           
            # Lecture CSV
            try:
                df = pd.read_csv(csv_path, header=None)
                for attr in target_attributes:
                    # On cherche la ligne correspondant à l'attribut
                    row = df[df.iloc[:, 0] == attr]
                    if not row.empty:
                        val = int(row.iloc[0, 1])
                        self.labels_dict[attr].append(val)
                    else:
                        self.labels_dict[attr].append(-1) # Fallback
            except Exception:
                pass
           
            count += 1
            if count >= SAMPLE_LIMIT: # On ne charge pas tout pour aller vite
                break
               
        # On garde seulement les images qu'on a chargées
        self.images = self.images[:count]
        print(f"Dataset prêt avec {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
       
        if self.transform:
            img = self.transform(img)
           
        # On retourne un dict de labels pour être flexible
        labels = {attr: self.labels_dict[attr][idx] for attr in self.target_attributes}
        return img, labels

# --- 3. CHARGEMENT DONNÉES ET MODÈLE ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# On utilise notre classe locale
dataset = CartoonProbeDataset(DATA_DIR, PROBE_ATTRIBUTES, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False) # Shuffle False pour garder l'ordre

# Setup du Modèle
# Vérifie bien que ces dimensions correspondent à ton entraînement !
model = CCVAE(
    img_channels=3,
    z_c_dim=16,
    z_not_c_dim=64,
    num_classes=10
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Poids du modèle chargés.")
except Exception as e:
    print(f"⚠️ Erreur chargement poids: {e}")

model.eval()

# --- 4. EXTRACTION DES VECTEURS Z_C ---
print("Extraction des vecteurs Latents (z_c)...")
X_zc = []
y_stored = {attr: [] for attr in PROBE_ATTRIBUTES}

with torch.no_grad():
    for x, labels_batch in loader:
        x = x.to(DEVICE)
       
        # Encodage
        h = model.encoder_conv(x)
        mu = model.fc_mu(h)
        z_c = mu[:, :model.z_c_dim].cpu().numpy() # On prend juste la partie supervisée
       
        X_zc.append(z_c)
       
        # Stockage des labels
        for attr in PROBE_ATTRIBUTES:
            y_stored[attr].extend(labels_batch[attr].numpy())

X_zc = np.concatenate(X_zc)
print(f"Extraction terminée. Shape: {X_zc.shape}")

# --- 5. AUDIT (CLASSIFICATION) ---
print("\n" + "="*80)
print(f"{'ATTRIBUT':<15} | {'ACCURACY':<10} | {'BASELINE':<10} | {'STATUS'}")
print("="*80)

for attr in PROBE_ATTRIBUTES:
    y_target = np.array(y_stored[attr])
   
    # --- PRÉTRAITEMENT SPECIFIQUE ---
    if attr == "glasses":
        # Binaire : 11 = Pas de lunettes, Reste = Lunettes
        # On veut prédire "A des lunettes" (1) vs "N'en a pas" (0)
        y_target = (y_target != 11).astype(int)
       
        # Baseline = Prédire la majorité
        freq = y_target.mean()
        baseline = max(freq, 1-freq)
       
    else:
        # Multiclasse standard
        vals, counts = np.unique(y_target, return_counts=True)
        baseline = np.max(counts) / np.sum(counts)

    # --- ENTRAINEMENT PROBE ---
    # Régression Logistique sur z_c
    X_train, X_test, y_train, y_test = train_test_split(X_zc, y_target, test_size=0.2, random_state=42)
   
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
   
    # --- DIAGNOSTIC ---
    if attr == "hair_color":
        # On veut que ce soit HAUT (c'est la tâche supervisée)
        status = "✅ OK (Supervisé)" if acc > 0.8 else "⚠️ Faible"
    else:
        # On veut que ce soit BAS (proche de la baseline)
        # Si acc est beaucoup plus haut que la baseline, c'est de la fuite (Leakage)
        threshold = baseline + 0.10 # Marge de 10%
        status = "❌ LEAKAGE" if acc > threshold else "✅ DISENTANGLED"

    print(f"{attr:<15} | {acc:.1%}    | {baseline:.1%}    | {status}")

print("="*80)