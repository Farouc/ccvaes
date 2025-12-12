# train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------
from model import CCVAE
from dataset import CartoonHairColorDataset

from losses.elbo_ccvae import (
    ccvae_elbo_supervised,
    ccvae_elbo_unsupervised,
)
from losses.ccvae_total_loss import ccvae_supervised_loss

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 70
LABELED_RATIO = 0.5
K_SAMPLES = 10

# Loss weights
ALPHA = 1.0    # unsupervised ELBO weight
GAMMA = 20.0   # auxiliary classification loss weight

# ------------------------------------------------------------
# 2. LOSS SELECTION
# ------------------------------------------------------------
LOSS_MODE = "ccvae_elbo"
# Options:
#   "ccvae_elbo"
#   "ccvae_contrastive"

CONTRASTIVE_WEIGHT = 5.0   # used only if ccvae_contrastive
CONTRASTIVE_ON = "mu"      # "mu" or "z" (currently only "mu" implemented)

# ------------------------------------------------------------
# 3. UTILS
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    return random_split(
        dataset,
        [n_labeled, n_unlabeled],
        generator=torch.Generator().manual_seed(42),
    )


def compute_supervised_loss(model, x, y):
    """
    Central dispatcher for supervised loss.
    """
    if LOSS_MODE == "ccvae_elbo":
        elbo, stats = ccvae_elbo_supervised(
            model,
            x,
            y,
            K=K_SAMPLES,
            recon_type="bce",
        )
        loss = -elbo
        stats["loss"] = loss.item()
        stats["loss_contrastive"] = 0.0
        return loss, stats

    elif LOSS_MODE == "ccvae_contrastive":
        return ccvae_supervised_loss(
            model,
            x,
            y,
            K=K_SAMPLES,
            recon_type="bce",
            contrastive_weight=CONTRASTIVE_WEIGHT,
            contrastive_on=CONTRASTIVE_ON,
        )

    else:
        raise ValueError(f"Unknown LOSS_MODE: {LOSS_MODE}")


# ------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results", exist_ok=True)

    # --- A. Data loading ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Loading CartoonSet10k...")
    dataset = CartoonHairColorDataset(
        root_dir="../data/cartoonset10k/cartoonset10k",
        transform=transform,
    )
    num_classes = dataset.num_classes

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"--> {len(labeled_set)} labeled | {len(unlabeled_set)} unlabeled")

    labeled_loader = DataLoader(
        labeled_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # --- B. Model ---
    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=64,
        num_classes=num_classes,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE}")
    print(f"LOSS_MODE={LOSS_MODE}, Gamma={GAMMA}, Alpha={ALPHA}")

    # --- C. Epoch loop ---
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_sup = 0.0
        total_unsup = 0.0
        total_class = 0.0
        total_contrast = 0.0
        total_acc = 0.0
        total_batches = 0

        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)

            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # --------------------------------------------------
            # Supervised loss
            # --------------------------------------------------
            loss_sup, stats_sup = compute_supervised_loss(
                model, x_labeled, y_labeled
            )

            # --------------------------------------------------
            # Unsupervised ELBO
            # --------------------------------------------------
            elbo_unsup, _ = ccvae_elbo_unsupervised(
                model,
                x_unlabeled,
                K=K_SAMPLES,
                recon_type="bce",
            )
            loss_unsup = -elbo_unsup

            # --------------------------------------------------
            # Auxiliary classification loss
            # --------------------------------------------------
            h = model.encoder_conv(x_labeled)
            mu = model.fc_mu(h)
            z_c = mu[:, :model.z_c_dim]
            logits = model.classifier(z_c)

            aux_class_loss = F.cross_entropy(logits, y_labeled)

            # --------------------------------------------------
            # Total loss
            # --------------------------------------------------
            loss = (
                loss_sup
                + ALPHA * loss_unsup
                + GAMMA * aux_class_loss
            )

            loss.backward()
            optimizer.step()

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_labeled).float().mean()

            total_loss += loss.item()
            total_sup += loss_sup.item()
            total_unsup += loss_unsup.item()
            total_class += aux_class_loss.item()
            total_contrast += stats_sup.get("loss_contrastive", 0.0)
            total_acc += acc.item()
            total_batches += 1

            if total_batches % 50 == 0:
                print(
                    f"[Batch {total_batches}] "
                    f"Loss={loss.item():.2f} | "
                    f"Sup={loss_sup.item():.2f} | "
                    f"Contrast={stats_sup.get('loss_contrastive', 0.0):.3f} | "
                    f"Acc={acc.item():.1%}"
                )

        # --------------------------------------------------
        # End of epoch
        # --------------------------------------------------
        print(f"\n===> Epoch {epoch+1}/{EPOCHS}")
        print(f" Avg Loss        : {total_loss / total_batches:.3f}")
        print(f" Avg Sup Loss    : {total_sup / total_batches:.3f}")
        print(f" Avg Unsup Loss  : {total_unsup / total_batches:.3f}")
        print(f" Avg Class Loss  : {total_class / total_batches:.3f}")
        print(f" Avg Contrastive : {total_contrast / total_batches:.3f}")
        print(f" Accuracy        : {total_acc / total_batches:.2%}")

        # --------------------------------------------------
        # Reconstructions
        # --------------------------------------------------
        with torch.no_grad():
            test_x = x_unlabeled[:8]
            recon_x, *_ = model(test_x)
            comparison = torch.cat([test_x, recon_x])
            save_image(
                comparison.cpu(),
                f"results/recon_epoch_{epoch+1}.png",
                nrow=8,
            )

        torch.save(model.state_dict(), "ccvae_haircolor.pth")
        print(" Model saved.\n")


if __name__ == "__main__":
    train()
