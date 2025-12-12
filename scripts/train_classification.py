import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import os
import sys

# Add root project path to system path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ccvae import CCVAE
from src.dataset.cartoonset import CartoonDataset
from src.loss import ccvae_loss_paper_supervised, ccvae_loss_paper_unsupervised

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Task Config
ATTRIBUTES = ["hair_color", "face_color"] 
IMG_SIZE = 64
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 70
LABELED_RATIO = 0.5   
K_SAMPLES = 10        

# Model Dimensions
Z_C_DIMS = [16, 16]   # [Hair, Face]
Z_NOT_C_DIM = 32

# Loss Weights
ALPHA = 1.0           # Unsupervised Weight
GAMMA = 20.0          # Auxiliary Classification Weight
CONTRASTIVE_WEIGHT = 5.0 

def split_supervised(dataset, labeled_ratio):
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    return random_split(dataset, [n_labeled, n_unlabeled], 
                        generator=torch.Generator().manual_seed(42))

def train():
    os.makedirs("results/classification", exist_ok=True)

    # --- A. Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print(f"Loading CartoonSet for attributes: {ATTRIBUTES}...")
    # Update path to where you stored the data
    dataset = CartoonDataset(
        root_dir="data/cartoonset10k", 
        target_attributes=ATTRIBUTES,
        transform=transform
    )
    
    num_classes_list = dataset.num_classes_list
    print(f"Classes per attribute: {num_classes_list}")

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> Data: {len(labeled_set)} Labeled | {len(unlabeled_set)} Unlabeled")

    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- B. Model Setup ---
    model = CCVAE(
        img_channels=3,
        img_size=IMG_SIZE,
        z_c_dims=Z_C_DIMS,
        z_not_c_dim=Z_NOT_C_DIM,
        task_types=['classification', 'classification'],
        num_classes_or_dim=num_classes_list,
        dropout_p=0.25 # Disentanglement trick
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Starting training on {DEVICE}...")

    # --- C. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_sup = 0
        total_unsup = 0
        total_aux = 0
        total_acc_per_attr = {i: 0.0 for i in range(len(ATTRIBUTES))}
        total_batches = 0

        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # Get Labeled Batch
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # 1. Supervised Loss (Paper Eq 4 + Contrastive)
            loss_sup, stats = ccvae_loss_paper_supervised(
                model, x_labeled, y_labeled, 
                K=K_SAMPLES, recon_type="bce",
                contrastive_weight=CONTRASTIVE_WEIGHT
            )

            # 2. Unsupervised Loss (Paper Eq 5)
            loss_unsup, _ = ccvae_loss_paper_unsupervised(
                model, x_unlabeled, 
                K=K_SAMPLES, recon_type="bce"
            )

            # 3. Auxiliary Loss (Gamma Boost)
            # Forward pass to get logits (preds)
            _, _, _, preds_list, _, _ = model(x_labeled)
            aux_loss = 0
            for i in range(len(ATTRIBUTES)):
                aux_loss += F.cross_entropy(preds_list[i], y_labeled[:, i])

            # Total Loss
            loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_loss)

            loss.backward()
            optimizer.step()

            # Monitoring
            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_unsup += loss_unsup.item()
            total_aux += aux_loss.item()
            total_batches += 1

            with torch.no_grad():
                for i in range(len(ATTRIBUTES)):
                    preds = torch.argmax(preds_list[i], dim=1)
                    acc = (preds == y_labeled[:, i]).float().mean()
                    total_acc_per_attr[i] += acc.item()

            if total_batches % 50 == 0:
                acc_str = " | ".join([f"{ATTRIBUTES[i]}: {(total_acc_per_attr[i]/total_batches):.1%}" for i in range(len(ATTRIBUTES))])
                print(f"   Batch {total_batches} | Loss: {loss.item():.1f} | Aux: {aux_loss.item():.2f} | {acc_str}")

        # End of Epoch
        avg_loss = total_loss / total_batches
        print(f"===> Epoch {epoch+1}/{EPOCHS} Finished. Avg Loss: {avg_loss:.2f}")

        # Save Checkpoint
        torch.save(model.state_dict(), "results/classification/ccvae_cartoon.pth")

        # Save Reconstructions
        with torch.no_grad():
            test_x = x_unlabeled[:8]
            recon_x, _, _, _, _, _ = model(test_x)
            comparison = torch.cat([test_x, recon_x])
            save_image(comparison.cpu(), f"results/classification/recon_epoch_{epoch+1}.png", nrow=8)

if __name__ == "__main__":
    train()