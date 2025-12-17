# train_fast_monolabel.py
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

# ------------------------------------------------------------
# Local imports (UNCHANGED)
# ------------------------------------------------------------
from model import CCVAE
from dataset import CartoonHairColorDataset
from losses.elbo_ccvae import (
    ccvae_elbo_supervised,
    ccvae_elbo_unsupervised,
)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128            # ⬆️ bigger batch
LR = 1e-4
EPOCHS = 500
LABELED_RATIO = 0.5
K_SAMPLES = 10

ALPHA = 1.0
GAMMA = 5.0

SAVE_EVERY = 5              # ⬇️ disk I/O

# ------------------------------------------------------------
# 2. UTILS
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

# ------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Loading dataset...")
    dataset = CartoonHairColorDataset(
        root_dir="data/cartoonset10k/cartoonset10k",
        transform=transform,
    )
    num_classes = dataset.num_classes

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)

    loader_kwargs = (
        dict(num_workers=4, pin_memory=True, persistent_workers=True)
        if DEVICE.type == "cuda"
        else {}
    )

    labeled_loader = DataLoader(
        labeled_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=64,
        num_classes=num_classes,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE}")

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        labeled_iter = iter(labeled_loader)

        for x_u, _ in unlabeled_loader:
            x_u = x_u.to(DEVICE)

            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            x_l = x_l.to(DEVICE)
            y_l = y_l.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # --------------------------------------------------
            # Supervised ELBO (unchanged)
            # --------------------------------------------------
            elbo_sup, _ = ccvae_elbo_supervised(
                model,
                x_l,
                y_l,
                K=K_SAMPLES,
                recon_type="bce",
            )
            loss_sup = -elbo_sup

            # --------------------------------------------------
            # Unsupervised ELBO (unchanged)
            # --------------------------------------------------
            elbo_unsup, _ = ccvae_elbo_unsupervised(
                model,
                x_u,
                K=K_SAMPLES,
                recon_type="bce",
            )
            loss_unsup = -elbo_unsup

            # --------------------------------------------------
            # Auxiliary classifier (UNAVOIDABLE extra pass)
            # --------------------------------------------------
            h = model.encoder_conv(x_l)
            mu = model.fc_mu(h)
            z_c = mu[:, :model.z_c_dim]
            logits = model.classifier(z_c)

            aux_class_loss = F.cross_entropy(logits, y_l)

            # --------------------------------------------------
            # Total loss
            # --------------------------------------------------
            loss = loss_sup + ALPHA * loss_unsup + GAMMA * aux_class_loss
            loss.backward()
            optimizer.step()

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y_l).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            batches += 1

        duration = time.time() - start
        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {total_loss/batches:.2f} | "
            f"Acc: {total_acc/batches:.2%} | "
            f"{duration:.1f}s"
        )

        # --------------------------------------------------
        # Save occasionally
        # --------------------------------------------------
        if (epoch + 1) % SAVE_EVERY == 0:
            with torch.no_grad():
                recon, *_ = model(x_u[:8])
                comparison = torch.cat([x_u[:8], recon])
                save_image(
                    comparison.cpu(),
                    f"results/recon_epoch_{epoch+1}.png",
                    nrow=8,
                )
            torch.save(model.state_dict(), "ccvae_monolabel_fast.pth")
            print("Checkpoint saved.")

if __name__ == "__main__":
    train()
