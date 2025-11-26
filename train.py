import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os

# Tes imports locaux (Mise à jour du nom du Dataset)
from model import CCVAE
from dataset import CartoonMultiLabelDataset 
from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attributs cibles
ATTRIBUTES = ["hair_color", "face_color"] 

# Hyperparamètres d'entraînement
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 70
LABELED_RATIO = 0.5 
K_SAMPLES = 10 

# Dimensions Latentes
# On donne 16 dimensions pour encoder les cheveux, 16 pour le visage
Z_C_DIMS = [16, 16] 
Z_NOT_C_DIM = 32 # On réduit un peu z_not_c pour forcer l'info dans z_c

# Hyperparamètres de Loss
ALPHA = 1.0  
GAMMA = 20.0 # Boost classification

# ------------------------------------------------------------
# 2. UTILS
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    return random_split(dataset, [n_labeled, n_unlabeled], 
                        generator=torch.Generator().manual_seed(42))

# ------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results_multi", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print(f"Chargement du dataset pour attributs : {ATTRIBUTES}...")
    
    # --- Changement ici : CartoonMultiLabelDataset ---
    dataset = CartoonMultiLabelDataset(
        root_dir="./cartoonset10k/cartoonset10k", 
        target_attributes=ATTRIBUTES,
        transform=transform
    )
    
    # On récupère la liste des classes (ex: [10, 11])
    num_classes_list = dataset.num_classes_list
    print(f"Classes par attribut : {num_classes_list}")

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> Données : {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- B. Model Setup (Multi-Label) ---
    model = CCVAE(
        img_channels=3,
        z_c_dims=Z_C_DIMS,       # Liste [16, 16]
        z_not_c_dim=Z_NOT_C_DIM, 
        num_classes_list=num_classes_list # Liste [10, 11]
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Démarrage de l'entraînement sur {DEVICE}...")

    # --- C. Epoch Loop ---
    for epoch in range(EPOCHS):
        model.train()
        
        # Stats cumulées
        total_loss = 0
        total_sup = 0
        total_unsup = 0
        total_aux = 0
        
        # Pour stocker l'accuracy par attribut (dictionnaire)
        total_acc_per_attr = {i: 0.0 for i in range(len(ATTRIBUTES))}
        
        total_batches = 0

        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # 1. Batch Labeled
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE) # Shape (B, 2)

            optimizer.zero_grad()

            # ---------------------------------------------------------
            # 2. Calcul des Loss
            # ---------------------------------------------------------

            # A. Loss Supervisée (Loss Paper Multi-Label)
            loss_sup, _ = ccvae_loss_supervised_paper(
                model, x_labeled, y_labeled, K=K_SAMPLES, recon_type="bce"
            )

            # B. Loss Non-Supervisée
            loss_unsup, _ = ccvae_loss_unsupervised_paper(
                model, x_unlabeled, K=K_SAMPLES, recon_type="bce"
            )

            # C. Auxiliary Classification Loss (Le Boost)
            # On doit faire un forward manuel pour récupérer les logits
            # model retourne : recon, mu, logvar, logits_list, prior_mu_list, prior_logvar_list
            _, _, _, logits_list, _, _ = model(x_labeled)
            
            aux_class_loss = 0
            
            # On somme la CrossEntropy pour chaque attribut
            # y_labeled[:, i] contient les labels pour l'attribut i
            for i in range(len(ATTRIBUTES)):
                aux_class_loss += F.cross_entropy(logits_list[i], y_labeled[:, i])

            # ---------------------------------------------------------
            # 3. Optimisation
            # ---------------------------------------------------------
            loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_class_loss)

            loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # 4. Monitoring
            # ---------------------------------------------------------
            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_unsup += loss_unsup.item()
            total_aux += aux_class_loss.item()
            total_batches += 1

            # Calcul Accuracy par attribut
            with torch.no_grad():
                for i in range(len(ATTRIBUTES)):
                    preds = torch.argmax(logits_list[i], dim=1)
                    acc = (preds == y_labeled[:, i]).float().mean()
                    total_acc_per_attr[i] += acc.item()

            if total_batches % 50 == 0:
                # On formate l'affichage des acc
                acc_str = " | ".join([f"{ATTRIBUTES[i]}: {(total_acc_per_attr[i]/total_batches):.1%}" 
                                      for i in range(len(ATTRIBUTES))])
                
                print(f"   Batch {total_batches} | Loss: {loss.item():.1f} | Aux: {aux_class_loss.item():.2f} | {acc_str}")

        # --- Fin de l'époque ---
        avg_loss = total_loss / total_batches
        avg_aux = total_aux / total_batches
        
        print(f"===> Epoch {epoch+1}/{EPOCHS} terminée.")
        print(f"     Avg Loss: {avg_loss:.2f} | Avg Aux Class Loss: {avg_aux:.4f}")
        
        for i in range(len(ATTRIBUTES)):
            avg_acc = total_acc_per_attr[i] / total_batches
            print(f"     Accuracy {ATTRIBUTES[i]}: {avg_acc:.2%}")

        # Sauvegarde images test
        with torch.no_grad():
            test_x = x_unlabeled[:8]
            # model(x) retourne un tuple, le premier élément est recon_x
            recon_x = model(test_x)[0] 
            comparison = torch.cat([test_x, recon_x])
            save_image(comparison.cpu(), f"results_multi/recon_epoch_{epoch+1}.png", nrow=8)

        torch.save(model.state_dict(), "ccvae_multilabel.pth")
        print("     Modèle sauvegardé.\n")

if __name__ == "__main__":
    train()