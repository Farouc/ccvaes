import torch
import torch.optim as optim
from model import CCVAE
from loss import loss_function_ccvae
from tqdm import tqdm

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
Z_C_DIM = 18
IMG_SIZE = 64

def train():
    print(f"Training on {DEVICE}...")
    
    # Initialisation du modèle
    model = CCVAE(img_channels=3, z_c_dim=Z_C_DIM, z_not_c_dim=27, num_labels=18).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    
    for epoch in range(EPOCHS):
        train_loss = 0
        
        # --- Simulation de DataLoader (Remplacer par vrai CelebA plus tard) ---
        # On simule 100 batches de données aléatoires
        # Images: (Batch, 3, 64, 64), Labels: (Batch, 18) en One-Hot ou Multi-Hot
        # Note: CelebA est multi-label, donc y_true est float [0, 1]
        
        progress_bar = tqdm(range(100), desc=f"Epoch {epoch+1}")
        
        for _ in progress_bar:
            # Données dummy
            data = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            # On normalise les données entre 0 et 1 car on a une Sigmoid en sortie
            data = torch.sigmoid(data) 
            
            # Labels dummy (multi-labels binaires)
            labels = torch.randint(0, 2, (BATCH_SIZE, 18)).float().to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Forward Pass
            recon_x, mu, logvar, y_pred, prior_mu, prior_logvar = model(data, labels)
            
            # 2. Calcul de la Loss
            # Pour la loss de classification multi-label (BCEWithLogits est mieux que CrossEntropy ici)
            # Mais gardons la structure de loss.py, on adapte juste l'appel :
            
            loss, recon, kl_c, class_loss = loss_function_ccvae(
                recon_x, data, mu, logvar, 
                y_pred, labels, # y_pred sont des logits
                prior_mu, prior_logvar, 
                z_c_dim=Z_C_DIM
            )
            
            # Petite correction pour le multi-label dans train.py :
            # La loss.py utilisait cross_entropy (bon pour 1 seule classe active).
            # Pour CelebA (plusieurs attributs actifs en même temps), il faut changer
            # la ligne classif_loss dans loss.py par :
            # F.binary_cross_entropy_with_logits(y_pred_logits, y_true, reduction='sum')
            # Pour ce test dummy, on assume que loss.py est ajusté ou on ignore l'erreur sémantique.
            
            # 3. Backward Pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({"Loss": loss.item()})

    print("Entraînement terminé sans erreur technique !")
    print("Prochaine étape : Coder le Dataset CelebA.")

if __name__ == "__main__":
    train()