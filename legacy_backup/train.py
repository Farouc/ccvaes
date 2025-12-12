# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os

# Imports locaux
from model import CCVAE
from dataset import CartoonHairColorDataset
from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 70
LABELED_RATIO = 0.5   
K_SAMPLES = 10        

# Loss Hyperparameters
ALPHA = 1.0           # Unsupervised Loss Weight
GAMMA = 20.0          # Auxiliary Classification Weight
CONTRASTIVE_WEIGHT = 5.0 # <--- NEW: Weight for Supervised Contrastive Loss on z_c

# ------------------------------------------------------------
# 2. UTILS
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    """Sépare le dataset en deux parties fixes."""
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    return random_split(dataset, [n_labeled, n_unlabeled], 
                        generator=torch.Generator().manual_seed(42))

# ------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results", exist_ok=True)

    # --- A. Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Chargement du dataset CartoonSet10k...")
    dataset = CartoonHairColorDataset(root_dir="./cartoonset10k/cartoonset10k", transform=transform)
    num_classes = dataset.num_classes

    # Split
    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> Données : {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

    # Dataloaders
    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- B. Model Setup ---
    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=64, # Check if your model uses 32 or 64
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Démarrage de l'entraînement sur {DEVICE}...")
    print(f"Options : Gamma={GAMMA}, Contrastive={CONTRASTIVE_WEIGHT}, Recon=BCE, Alpha={ALPHA}")

    # --- C. Epoch Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_sup = 0
        total_class_loss = 0
        total_contrastive_loss = 0
        total_acc = 0
        total_batches = 0

        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # 1. Get Labeled Batch (Infinite Loop)
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # ---------------------------------------------------------
            # 2. Calculate Losses
            # ---------------------------------------------------------

            # A. Supervised Loss (CCVAE Eq 4 + Contrastive)
            # We pass the new contrastive_weight parameter here
            loss_sup, stats_sup = ccvae_loss_supervised_paper(
                model, x_labeled, y_labeled, K=K_SAMPLES, recon_type="bce",
                contrastive_weight=CONTRASTIVE_WEIGHT
            )

            # B. Unsupervised Loss (CCVAE Eq 5)
            loss_unsup, _ = ccvae_loss_unsupervised_paper(
                model, x_unlabeled, K=K_SAMPLES, recon_type="bce"
            )

            # C. Auxiliary Classification Loss (Gamma Boost)
            h = model.encoder_conv(x_labeled)
            mu = model.fc_mu(h)
            z_c = mu[:, :model.z_c_dim]
            logits = model.classifier(z_c)
            
            aux_class_loss = F.cross_entropy(logits, y_labeled)

            # ---------------------------------------------------------
            # 3. Optimization
            # ---------------------------------------------------------
            
            # Note: loss_sup already includes the weighted contrastive term
            loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_class_loss)

            loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # 4. Monitoring
            # ---------------------------------------------------------
            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_class_loss += aux_class_loss.item()
            # Retrieve contrastive loss from stats dict
            total_contrastive_loss += stats_sup.get('loss_contrastive', 0.0)
            total_batches += 1

            # Calculate Accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_labeled).float().mean()
                total_acc += acc.item()

            if total_batches % 50 == 0:
                print(f"   [Batch {total_batches}] Loss: {loss.item():.1f} | "
                      f"Sup: {loss_sup.item():.1f} | "
                      f"Contrast: {stats_sup.get('loss_contrastive', 0.0):.3f} | "
                      f"Acc: {acc.item():.1%}")

        # --- End of Epoch ---
        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        avg_class_loss = total_class_loss / total_batches
        avg_contrast_loss = total_contrastive_loss / total_batches
        
        print(f"===> Epoch {epoch+1}/{EPOCHS} terminée.")
        print(f"     Avg Loss: {avg_loss:.2f} | Avg Class Loss: {avg_class_loss:.4f} | Avg Contrast Loss: {avg_contrast_loss:.4f}")
        print(f"     Global Accuracy: {avg_acc:.2%}")

        # Save Test Images
        with torch.no_grad():
            test_x = x_unlabeled[:8]
            recon_x, _, _, _, _, _ = model(test_x)
            comparison = torch.cat([test_x, recon_x])
            save_image(comparison.cpu(), f"results/recon_epoch_{epoch+1}.png", nrow=8)

        # Save Model
        torch.save(model.state_dict(), "ccvae_haircolor.pth")
        print("     Modèle sauvegardé.\n")

if __name__ == "__main__":
    train()

# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.utils import save_image
# import os

# # Tes imports locaux
# from model import CCVAE
# from dataset import CartoonHairColorDataset
# from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# # ------------------------------------------------------------
# # 1. CONFIGURATION
# # ------------------------------------------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparamètres d'entraînement
# BATCH_SIZE = 32        # 32 est plus stable pour la convergence que 64 au début
# LR = 1e-4              # Learning rate standard pour VAE
# EPOCHS = 70
# LABELED_RATIO = 0.5   # 20% de données étiquetées
# K_SAMPLES = 10         # Nombre d'échantillons MC (Importance Sampling)

# # Hyperparamètres de Loss
# ALPHA = 1.0            # Poids de la loss non-supervisée
# GAMMA = 20.0           # <--- LE BOOST : Force le modèle à classifier correctement tout de suite

# # ------------------------------------------------------------
# # 2. UTILS
# # ------------------------------------------------------------
# def split_supervised(dataset, labeled_ratio):
#     """Sépare le dataset en deux parties fixes."""
#     n_total = len(dataset)
#     n_labeled = int(n_total * labeled_ratio)
#     n_unlabeled = n_total - n_labeled
#     # Le seed 42 assure que ce sont toujours les mêmes images qui sont labeled
#     return random_split(dataset, [n_labeled, n_unlabeled], 
#                         generator=torch.Generator().manual_seed(42))

# # ------------------------------------------------------------
# # 3. TRAINING LOOP
# # ------------------------------------------------------------
# def train():
#     # Création du dossier de sauvegarde
#     os.makedirs("results", exist_ok=True)

#     # --- A. Data Loading ---
#     # Note: Pas de normalisation moyenne/std car BCE attend des inputs [0, 1]
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ])

#     print("Chargement du dataset CartoonSet10k...")
#     # Vérifie bien que le chemin pointe vers le dossier contenant les images
#     dataset = CartoonHairColorDataset(root_dir="./cartoonset10k/cartoonset10k", transform=transform)
#     num_classes = dataset.num_classes

#     # Split
#     labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
#     print(f"--> Données : {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

#     # Dataloaders (drop_last=True évite les crashs sur le dernier batch incomplet)
#     labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#     # --- B. Model Setup ---
#     model = CCVAE(
#         img_channels=3,
#         z_c_dim=16,
#         z_not_c_dim=64, # Doit matcher ton model.py (32 ou 64, vérifie ton fichier)
#         num_classes=num_classes
#     ).to(DEVICE)

#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     print(f"Démarrage de l'entraînement sur {DEVICE}...")
#     print(f"Options : Gamma={GAMMA}, Recon=BCE, Alpha={ALPHA}")

#     # --- C. Epoch Loop ---
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
#         total_sup = 0
#         total_class_loss = 0
#         total_acc = 0
#         total_batches = 0

#         # Itérateur pour la boucle infinie sur les données labeled
#         labeled_iter = iter(labeled_loader)

#         # On boucle sur le GRAND dataset (Unlabeled) pour voir toutes les données
#         for x_unlabeled, _ in unlabeled_loader:
#             x_unlabeled = x_unlabeled.to(DEVICE)

#             # 1. Récupération du batch labeled (Cycle infini)
#             try:
#                 x_labeled, y_labeled = next(labeled_iter)
#             except StopIteration:
#                 labeled_iter = iter(labeled_loader)
#                 x_labeled, y_labeled = next(labeled_iter)
            
#             x_labeled = x_labeled.to(DEVICE)
#             y_labeled = y_labeled.to(DEVICE)

#             optimizer.zero_grad()

#             # ---------------------------------------------------------
#             # 2. Calcul des Loss
#             # ---------------------------------------------------------

#             # A. Loss Supervisée (Maths du papier Eq 4)
#             # recon_type="bce" est crucial pour la netteté des images
#             loss_sup, _ = ccvae_loss_supervised_paper(
#                 model, x_labeled, y_labeled, K=K_SAMPLES, recon_type="bce"
#             )

#             # B. Loss Non-Supervisée (Maths du papier Eq 5)
#             loss_unsup, _ = ccvae_loss_unsupervised_paper(
#                 model, x_unlabeled, K=K_SAMPLES, recon_type="bce"
#             )

#             # C. Auxiliary Classification Loss (Le Boost Pratique)
#             # On force explicitement z_c à prédire la classe
#             h = model.encoder_conv(x_labeled)
#             mu = model.fc_mu(h)
#             z_c = mu[:, :model.z_c_dim]
#             logits = model.classifier(z_c)
            
#             aux_class_loss = F.cross_entropy(logits, y_labeled)

#             # ---------------------------------------------------------
#             # 3. Optimisation
#             # ---------------------------------------------------------
            
#             # Loss Totale = Maths VAE + (Gamma * Classification Explicite)
#             loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_class_loss)

#             loss.backward()
#             optimizer.step()

#             # ---------------------------------------------------------
#             # 4. Monitoring
#             # ---------------------------------------------------------
#             total_loss += loss.item()
#             total_sup += loss_sup.item()
#             total_class_loss += aux_class_loss.item()
#             total_batches += 1

#             # Calcul Accuracy (sur le batch labeled courant)
#             with torch.no_grad():
#                 preds = torch.argmax(logits, dim=1)
#                 acc = (preds == y_labeled).float().mean()
#                 total_acc += acc.item()

#             # Log fréquent (tous les 50 batches)
#             if total_batches % 50 == 0:
#                 print(f"   [Batch {total_batches}] Loss: {loss.item():.1f} | "
#                       f"Sup: {loss_sup.item():.1f} | "
#                       f"Class(Aux): {aux_class_loss.item():.3f} | "
#                       f"Acc: {acc.item():.1%}")

#         # --- Fin de l'époque ---
#         avg_loss = total_loss / total_batches
#         avg_acc = total_acc / total_batches
#         avg_class_loss = total_class_loss / total_batches
        
#         print(f"===> Epoch {epoch+1}/{EPOCHS} terminée.")
#         print(f"     Avg Loss: {avg_loss:.2f} | Avg Class Loss: {avg_class_loss:.4f}")
#         print(f"     Global Accuracy: {avg_acc:.2%}")

#         # Sauvegarde d'images de test (sur données non vues/non supervisées)
#         with torch.no_grad():
#             test_x = x_unlabeled[:8] # On prend 8 images du dernier batch unlabeled
#             recon_x, _, _, _, _, _ = model(test_x)
#             # Concaténation : Ligne du haut = Originale / Ligne du bas = Reconstruction
#             comparison = torch.cat([test_x, recon_x])
#             save_image(comparison.cpu(), f"results/recon_epoch_{epoch+1}.png", nrow=8)

#         # Sauvegarde du modèle
#         torch.save(model.state_dict(), "ccvae_haircolor.pth")
#         print("     Modèle sauvegardé.\n")

# if __name__ == "__main__":
#     train()
