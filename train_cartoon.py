# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import random

from model import CCVAE
from loss import ccvae_loss_supervised, ccvae_loss_unsupervised
from dataset import CartoonHairColorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20
PERCENT_LABELED = 0.70    # <-- ici tu choisis ton pourcentage
IMG_PATH = "./cartoonset10k/cartoonset10k"


# Accuracy softmax
def accuracy_softmax(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()


def train():

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Chargement dataset...")
    full_dataset = CartoonHairColorDataset(IMG_PATH, transform)
    num_classes = full_dataset.num_classes
    N = len(full_dataset)

    # ------------------------------------------
    # 1. SPLIT LABELLED / UNLABELLED
    # ------------------------------------------
    indices = list(range(N))
    random.shuffle(indices)
    n_labeled = int(PERCENT_LABELED * N)

    labeled_idx = indices[:n_labeled]
    unlabeled_idx = indices[n_labeled:]

    labeled_set = Subset(full_dataset, labeled_idx)
    unlabeled_set = Subset(full_dataset, unlabeled_idx)

    print(f"Labeled samples : {len(labeled_set)}")
    print(f"Unlabeled samples : {len(unlabeled_set)}")

    loader_l = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_u = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True)

    # ------------------------------------------
    # 2. INIT CCVAE
    # ------------------------------------------
    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=32,
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Démarrage de l'entraînement...\n")

    for epoch in range(EPOCHS):

        loop_l = iter(loader_l)
        loop_u = iter(loader_u)

        steps = max(len(loader_l), len(loader_u))
        total_loss = 0
        total_acc = 0
        count_labeled_batches = 0

        for step in range(steps):

            # -------------------------
            # 3A. BATCH LABELED
            # -------------------------
            try:
                x_l, y_l = next(loop_l)
                x_l = x_l.to(DEVICE)
                y_l = y_l.to(DEVICE)

                recon_l, mu_l, logvar_l, y_logits_l, prior_mu_l, prior_logvar_l = model(x_l, y_l)

                # Loss supervisée
                loss_l, rec_l, klnot_l, klc_l, cls_l = ccvae_loss_supervised(
                    recon_l, x_l,
                    mu_l, logvar_l,
                    y_logits_l, y_l,
                    prior_mu_l, prior_logvar_l,
                    z_c_dim=model.z_c_dim
                )

                acc_l = accuracy_softmax(y_logits_l, y_l)
                count_labeled_batches += 1

            except StopIteration:
                loss_l = 0
                acc_l = 0

            # -------------------------
            # 3B. BATCH UNLABELED
            # -------------------------
            try:
                x_u, _ = next(loop_u)
                x_u = x_u.to(DEVICE)

                recon_u, mu_u, logvar_u, _, _, _ = model(x_u, y=None)

                # Loss non supervisée
                loss_u, rec_u, kl_u = ccvae_loss_unsupervised(
                    recon_u, x_u,
                    mu_u, logvar_u,
                    z_c_dim=model.z_c_dim
                )

            except StopIteration:
                loss_u = 0

            # -------------------------
            # 3C. COMBINAISON
            # -------------------------
            total_batch_loss = loss_l + loss_u

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_acc += acc_l

            tqdm.write(
                f"[Epoch {epoch+1}] Step {step+1}/{steps} | "
                f"Loss={total_batch_loss.item():.3f} | "
                f"Acc={acc_l:.2%}"
            )

        print(f"===> Epoch {epoch+1}: Loss={total_loss/steps:.4f} | Acc={total_acc/max(count_labeled_batches,1):.2%}\n")

        torch.save(model.state_dict(), f"ccvae_haircolor_epoch{epoch+1}.pth")

    print("Entraînement terminé !")


if __name__ == "__main__":
    train()
