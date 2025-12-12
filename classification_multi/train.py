import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

# Local imports
from model import CCVAE
from dataset import CartoonMultiLabelDataset
from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target attributes for multi-label learning
ATTRIBUTES = ["hair_color", "face_color"]

# Training hyperparameters
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 110
LABELED_RATIO = 0.5
K_SAMPLES = 10

# Latent dimensions and loss weights
Z_C_DIMS = [16, 16]      # One z_c block per attribute
Z_NOT_C_DIM = 32        # Shared nuisance / style latent
GAMMA = 30.0            # Auxiliary classification weight
ALPHA = 1.0             # Unsupervised loss weight

# ------------------------------------------------------------
# 2. UTILS
# ------------------------------------------------------------
def split_supervised(dataset, labeled_ratio):
    """
    Split the dataset into labeled and unlabeled subsets.
    The split is deterministic for reproducibility.
    """
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

    print("Loading Cartoon multi-label dataset...")
    dataset = CartoonMultiLabelDataset(
        root_dir="../data/cartoonset10k/cartoonset10k",
        target_attributes=ATTRIBUTES,
        transform=transform,
    )

    num_classes_list = dataset.num_classes_list

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    print(f"Dataset split: {len(labeled_set)} labeled | {len(unlabeled_set)} unlabeled")

    # DataLoader performance options
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

    # Model
    model = CCVAE(
        img_channels=3,
        z_c_dims=Z_C_DIMS,
        z_not_c_dim=Z_NOT_C_DIM,
        num_classes_list=num_classes_list,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Starting training on {DEVICE} (checkpoint every 5 epochs)")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        total_aux_loss = 0.0
        total_batches = 0

        # Accuracy accumulators per attribute
        total_acc_per_attr = {i: 0.0 for i in range(len(ATTRIBUTES))}

        labeled_iter = iter(labeled_loader)

        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # Infinite cycling over labeled data
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)

            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # --------------------------------------------------
            # A. CCVAE losses (supervised + unsupervised)
            # --------------------------------------------------
            loss_sup, _ = ccvae_loss_supervised_paper(
                model,
                x_labeled,
                y_labeled,
                K=K_SAMPLES,
                recon_type="bce",
            )

            loss_unsup, _ = ccvae_loss_unsupervised_paper(
                model,
                x_unlabeled,
                K=K_SAMPLES,
                recon_type="bce",
            )

            # --------------------------------------------------
            # B. Auxiliary classification loss + accuracy
            # --------------------------------------------------
            _, _, _, logits_list, _, _ = model(x_labeled)

            aux_class_loss = 0.0

            for i in range(len(ATTRIBUTES)):
                aux_class_loss += F.cross_entropy(
                    logits_list[i],
                    y_labeled[:, i],
                )

                with torch.no_grad():
                    preds = torch.argmax(logits_list[i], dim=1)
                    acc = (preds == y_labeled[:, i]).float().mean()
                    total_acc_per_attr[i] += acc.item()

            # --------------------------------------------------
            # C. Total loss
            # --------------------------------------------------
            loss = loss_sup + (ALPHA * loss_unsup) + (GAMMA * aux_class_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_aux_loss += aux_class_loss.item()
            total_batches += 1

            if total_batches % 50 == 0:
                acc_str = " | ".join(
                    f"{ATTRIBUTES[i]}: {(total_acc_per_attr[i] / total_batches):.1%}"
                    for i in range(len(ATTRIBUTES))
                )
                print(
                    f"  Batch {total_batches} | "
                    f"Loss: {loss.item():.1f} | {acc_str}"
                )

        # --------------------------------------------------
        # End of epoch
        # --------------------------------------------------
        duration = time.time() - start_time
        avg_loss = total_loss / total_batches

        print(f"Epoch {epoch+1}/{EPOCHS} | {duration:.0f}s | Avg Loss: {avg_loss:.1f}")
        for i in range(len(ATTRIBUTES)):
            avg_acc = total_acc_per_attr[i] / total_batches
            print(f"  -> {ATTRIBUTES[i]} accuracy: {avg_acc:.2%}")

        # --------------------------------------------------
        # Checkpoint & visualization (every 5 epochs)
        # --------------------------------------------------
        if (epoch + 1) % 5 == 0:
            print("  Saving checkpoint and reconstructions...")
            with torch.no_grad():
                test_x = x_unlabeled[:8]
                recon_x = model(test_x)[0]
                comparison = torch.cat([test_x, recon_x])
                save_image(
                    comparison.cpu(),
                    f"results/recon_epoch_{epoch+1}.png",
                    nrow=8,
                )

            torch.save(model.state_dict(), "ccvae_multilabel_r.pth")


if __name__ == "__main__":
    train()
