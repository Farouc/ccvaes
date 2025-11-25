# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import CCVAE
from dataset import CartoonHairColorDataset
from loss import (
    ccvae_loss_supervised_paper,
    ccvae_loss_unsupervised_paper
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 30
LABELED_RATIO = 0.20   # pourcentage d’images avec labels
K_SAMPLES = 10         # importance sampling, K du papier


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    """
    Renvoie deux datasets : labeled_dataset, unlabeled_dataset
    """
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled

    labeled_set, unlabeled_set = random_split(
        dataset,
        [n_labeled, n_unlabeled],
        generator=torch.Generator().manual_seed(42)
    )

    return labeled_set, unlabeled_set


# ------------------------------------------------------------
# TRAIN LOOP
# ------------------------------------------------------------
def train():

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    print("Chargement du dataset CartoonSet10k...")
    dataset = CartoonHairColorDataset(root_dir="./cartoonset10k/cartoonset10k",
                                      transform=transform)

    num_classes = dataset.num_classes
    print(f"--> hair_color contient {num_classes} classes différentes.")

    # --------------------------------------------------------
    # Split labeled/unlabeled
    # --------------------------------------------------------
    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> {len(labeled_set)} images supervisées")
    print(f"--> {len(unlabeled_set)} images non supervisées")

    # DataLoaders
    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, drop_last=True)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=64,
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Démarrage entraînement CCVAE...")

    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0
        total_batches = 0

        # Synchroniser les itérateurs
        unlabeled_iter = iter(unlabeled_loader)

        for x_labeled, y_labeled in labeled_loader:
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            # prendre un batch non supervisé
            try:
                x_unlabeled, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_unlabeled, _ = next(unlabeled_iter)

            x_unlabeled = x_unlabeled.to(DEVICE)

            optimizer.zero_grad()

            # -------------------------------------------------
            # supervised loss (Eq 4)
            # -------------------------------------------------
            loss_sup, stats_sup = ccvae_loss_supervised_paper(
                model,
                x_labeled,
                y_labeled,
                K=K_SAMPLES,
                recon_type="mse"
            )

            # -------------------------------------------------
            # unsupervised loss (Eq 5)
            # -------------------------------------------------
            loss_unsup, stats_unsup = ccvae_loss_unsupervised_paper(
                model,
                x_unlabeled,
                K=K_SAMPLES,
                recon_type="mse"
            )

            # combinaison supervisé + non-supervisé
            loss = loss_sup + loss_unsup

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        print(f"[Epoch {epoch+1}/{EPOCHS}]  Loss = {avg_loss:.4f}")

        # ----------------------------------------------------
        # Save model regularly
        # ----------------------------------------------------
        torch.save(model.state_dict(), "ccvae_haircolor.pth")


    print("Entraînement terminé !")


if __name__ == "__main__":
    train()
