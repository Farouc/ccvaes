import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os
import time

# Imports spécifiques Régression
from model import CCVAE_Age
from dataset import UTKFaceDataset
from loss import loss_regression_paper

# ------------------------------------------------------------
# 1. CONFIGURATION REGRESSION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
BATCH_SIZE = 32       # 64 est bien pour des images 64x64
LR = 1e-4
EPOCHS = 200            # Ça converge assez vite
LABELED_RATIO = 0.8    # On a beaucoup de labels (tout UTKFace), autant en profiter !

# Dimensions
Z_C_DIM = 16           # Age embedding
Z_NOT_C_DIM = 64       # Identité (besoin de place pour les détails du visage)
GAMMA = 100           # Poids de la régression (MSE)
BETA=0.001

# ------------------------------------------------------------
# 2. UTILS
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    # On garde le seed pour la reproductibilité
    return random_split(dataset, [n_labeled, n_unlabeled], 
                        generator=torch.Generator().manual_seed(42))

# ------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results_age_128_128", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("Chargement du dataset UTKFace...")
    # Assure-toi que le chemin est bon !
    dataset = UTKFaceDataset(root_dir="./UTKFace", transform=transform)
    
    # Split Labeled / Unlabeled
    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> Données : {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True} if DEVICE.type == 'cuda' else {}
    
    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)

    # Modèle Régression
    model = CCVAE_Age(
        img_channels=3,
        z_c_dim=Z_C_DIM,
        z_not_c_dim=Z_NOT_C_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    print(f"Démarrage Entraînement AGE sur {DEVICE}...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        total_reg_loss = 0
        total_mae = 0
        total_batches = 0
        
        labeled_iter = iter(labeled_loader)

        # On boucle sur le dataset Unlabeled
        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # 1. Récupération Batch Labeled (Image + Age)
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE) # (B, 1) float

            optimizer.zero_grad()

            # --- Forward ---
            recon, mu, logvar, y_pred, p_mu, p_logvar = model(x_labeled, y_labeled)

            # --- Loss ---
            loss, stats = loss_regression_paper(
                recon, x_labeled, mu, logvar, 
                y_pred, y_labeled, 
                p_mu, p_logvar, 
                gamma=GAMMA,beta=BETA
            )

            # --- Backward ---
            loss.backward()
            optimizer.step()

            # --- Stats ---
            total_loss += loss.item()
            total_reg_loss += stats['reg'] 
            
            # Calcul MAE (Mean Absolute Error) en années (x100 car normalisé)
            with torch.no_grad():
                mae = torch.abs(y_pred - y_labeled).mean().item() * 100
                total_mae += mae

            total_batches += 1

            if total_batches % 50 == 0:
                print(f"   Batch {total_batches} | Loss: {loss.item():.1f} | MAE: {mae:.1f} ans")

        # --- Fin d'époque ---
        duration = time.time() - start_time
        avg_loss = total_loss / total_batches
        avg_mae = total_mae / total_batches
        
        print(f"Epoch {epoch+1}/{EPOCHS} | {duration:.0f}s | Loss: {avg_loss:.1f} | MAE Global: {avg_mae:.2f} ans")
        scheduler.step(avg_mae)

        # --- VISUALISATION (Reconstruction + Aging) ---
        if (epoch + 1) % 5 == 0 or (epoch == 0):
            model.eval()
            print("    [Génération Visualisations...]")
            
            with torch.no_grad():
                # --- A. RECONSTRUCTION ---
                test_batch = x_unlabeled[:8]
                recon_batch, _, _, _, _, _ = model(test_batch)
                recon_comparison = torch.cat([test_batch, recon_batch])
                save_image(recon_comparison.cpu(), f"results_age_128_128/recon_epoch_{epoch+1}.png", nrow=8)

                # --- B. AGING STRIP ---
                target_img = x_unlabeled[0:1] 
                _, mu_enc, _, _, _, _ = model(target_img)
                z_not_c = mu_enc[:, Z_C_DIM:]
                
                # Ages cibles
                ages_steps = torch.linspace(0.1, 0.9, 9).unsqueeze(1).to(DEVICE)
                h_prior = model.prior_net(ages_steps)
                z_c_targets = model.prior_mu(h_prior)
                
                # Combine
                z_not_c_repeated = z_not_c.repeat(9, 1)
                z_combined = torch.cat([z_c_targets, z_not_c_repeated], dim=1)
                
                # --- CORRECTION ICI : view(-1, 64, 4, 4) ---
                # Cela correspond à 1024 neurones / (4*4) = 64 canaux
                # dec_in = model.decoder_input(z_combined).view(-1, 64, 4, 4) 
                dec_in = model.decoder_input(z_combined).view(-1, 512, 4, 4) 
                
                generated_faces = model.decoder_conv(dec_in)
                
                aging_strip = torch.cat([target_img, generated_faces])
                save_image(aging_strip.cpu(), f"results_age_128_128/aging_epoch_{epoch+1}.png", nrow=10)
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), "ccvae_regression_utk_128.pth")
            print("    [Images et Modèle sauvegardés]")

if __name__ == "__main__":
    train()