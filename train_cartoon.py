import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os 
from PIL import Image 
from tqdm import tqdm

# Tes fichiers
from model import CCVAE
from loss import loss_function_ccvae
from dataset import CartoonDataset 

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20 
IMG_PATH = "./cartoonset10k/cartoonset10k" # Ton chemin exact
# Note: Z_C_DIM est maintenant défini dynamiquement dans la fonction train()

def calculate_accuracy(y_pred_logits, y_true):
    probs = torch.sigmoid(y_pred_logits)
    preds = (probs > 0.5).float()
    correct = (preds == y_true).float().sum()
    total = y_true.numel() 
    return correct / total

def train():
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Chargement des données...")
    # MODIFICATION 1: On utilise le nouveau constructeur (sans csv_file)
    dataset = CartoonDataset(root_dir=IMG_PATH, transform=transform, num_images=10000)
    
    # MODIFICATION 2: num_workers=0 pour la stabilité sous Windows
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # MODIFICATION 3: Auto-détection des dimensions
    test_img, test_label = dataset[0]
    num_attributes = len(test_label)
    print(f"--> Nombre d'attributs détectés par image : {num_attributes}")
    
    # On fixe la dimension latente caractéristique égale au nombre d'attributs
    z_c_dim_dynamic = num_attributes

    # Init du modèle avec les dimensions dynamiques
    model = CCVAE(
        img_channels=3, 
        z_c_dim=z_c_dim_dynamic, 
        z_not_c_dim=32, 
        num_labels=num_attributes
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    loss_history = []
    acc_history = []

    model.train()
    print(f"Début entraînement sur {DEVICE} avec Z_C_DIM={z_c_dim_dynamic}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        steps = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            recon_x, mu, logvar, y_pred, prior_mu, prior_logvar = model(imgs, labels)
            
            # Loss
            loss, recon, kl, classif = loss_function_ccvae(
                recon_x, imgs, mu, logvar, y_pred, labels, 
                prior_mu, prior_logvar, z_c_dim=z_c_dim_dynamic
            )
            
            # Accuracy
            acc = calculate_accuracy(y_pred, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            steps += 1
            
            loop.set_postfix(Loss=loss.item(), Acc=f"{acc.item():.2%}")

        avg_loss = total_loss / steps
        avg_acc = total_acc / steps
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        print(f"Epoch {epoch+1} terminé. Loss: {avg_loss:.4f} | Accuracy Classification: {avg_acc:.2%}")
        
        # Sauvegarder le modèle
        torch.save(model.state_dict(), "ccvae_cartoon.pth")

    print("Entraînement terminé.")

if __name__ == "__main__":
    # Création dossier dummy si inexistant (sécurité)
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)
        try:
            Image.new('RGB', (64, 64)).save(os.path.join(IMG_PATH, "dummy.jpg"))
        except:
            pass
        
    train()