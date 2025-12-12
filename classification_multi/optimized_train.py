import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os
import time

# Tes imports locaux
from model import CCVAE
from dataset import CartoonMultiLabelDataset 
from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTRIBUTES = ["hair_color", "face_color"] 

# Hyperparamètres
BATCH_SIZE = 128       
LR = 1e-4
EPOCHS = 110
LABELED_RATIO = 0.5 
K_SAMPLES = 10         

# Dimensions & Poids
Z_C_DIMS = [16, 16] 
Z_NOT_C_DIM = 32       
GAMMA = 30 
ALPHA = 1.0

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
    os.makedirs("results_multi_round3", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Chargement du dataset...")
    dataset = CartoonMultiLabelDataset(
        root_dir="./cartoonset10k/cartoonset10k", 
        target_attributes=ATTRIBUTES,
        transform=transform
    )
    num_classes_list = dataset.num_classes_list
    
    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> Données : {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

    # Optimisation CPU/GPU (Workers)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True} if DEVICE.type == 'cuda' else {}
    
    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)

    model = CCVAE(
        img_channels=3,
        z_c_dims=Z_C_DIMS,
        z_not_c_dim=Z_NOT_C_DIM, 
        num_classes_list=num_classes_list
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Démarrage TURBO sur {DEVICE} (Save every 10 epochs)...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        total_aux = 0
        total_batches = 0
        
        # Accumulateurs pour l'accuracy
        total_acc_per_attr = {i: 0.0 for i in range(len(ATTRIBUTES))}
        
        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # Cycle infini Labeled
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # Loss A & B (K=1)
            loss_sup, _ = ccvae_loss_supervised_paper(model, x_labeled, y_labeled, K=K_SAMPLES, recon_type="bce")
            loss_unsup, _ = ccvae_loss_unsupervised_paper(model, x_unlabeled, K=K_SAMPLES, recon_type="bce")

            # Loss C (Auxiliary) + Calcul Accuracy
            _, _, _, logits_list, _, _ = model(x_labeled)
            
            aux_class_loss = 0
            
            for i in range(len(ATTRIBUTES)):
                # 1. Loss de classification
                aux_class_loss += F.cross_entropy(logits_list[i], y_labeled[:, i])
                
                # 2. Calcul Accuracy (Log)
                with torch.no_grad():
                    preds = torch.argmax(logits_list[i], dim=1)
                    acc = (preds == y_labeled[:, i]).float().mean()
                    total_acc_per_attr[i] += acc.item()

            # Total Loss
            loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_class_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_aux += aux_class_loss.item()
            total_batches += 1

            # Log fréquent (tous les 50 batches)
            if total_batches % 50 == 0:
                acc_str = " | ".join([f"{ATTRIBUTES[i]}: {(total_acc_per_attr[i]/total_batches):.1%}" 
                                      for i in range(len(ATTRIBUTES))])
                print(f"   Batch {total_batches} | Loss: {loss.item():.1f} | {acc_str}")

        # --- Fin de l'époque ---
        duration = time.time() - start_time
        avg_loss = total_loss / total_batches
        
        print(f"Epoch {epoch+1}/{EPOCHS} | {duration:.0f}s | Avg Loss: {avg_loss:.1f}")
        for i in range(len(ATTRIBUTES)):
            avg_acc = total_acc_per_attr[i] / total_batches
            print(f"    -> {ATTRIBUTES[i]} Accuracy: {avg_acc:.2%}")

        # --- SAUVEGARDE : UNIQUEMENT TOUTES LES 10 ÉPOQUES ---
        if (epoch + 1) % 5 == 0:
            print("    [Sauvegarde Checkpoint...]")
            with torch.no_grad():
                test_x = x_unlabeled[:8]
                recon_x = model(test_x)[0] 
                comparison = torch.cat([test_x, recon_x])
                save_image(comparison.cpu(), f"results_multi_round3/recon_epoch_{epoch+1}.png", nrow=8)
            
            torch.save(model.state_dict(), "ccvae_multilabel_round3.pth")

if __name__ == "__main__":
    train()