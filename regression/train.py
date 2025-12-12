import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image

# Regression-specific imports
from model import CCVAE_Age
from dataset import UTKFaceDataset
from loss import loss_regression_paper

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 200

# Latent dimensions
Z_C_DIM = 16        # age-related latent
Z_NOT_C_DIM = 64    # identity / nuisance latent

# Loss weights
GAMMA = 100.0       # regression weight
BETA = 0.001        # KL weight

# ------------------------------------------------------------
# 2. BALANCED SAMPLER
# ------------------------------------------------------------
def make_balanced_sampler(subset_indices, full_dataset):
    """
    Create a balanced sampler for the training subset based on age frequency.
    """
    print("Computing sampling weights for balanced training set...")

    ages = []
    for idx in subset_indices:
        img_name = full_dataset.image_files[idx]
        try:
            age = int(img_name.split("_")[0])
        except Exception:
            age = 25
        ages.append(age)

    ages = np.asarray(ages)

    # Histogram over ages (0–117)
    counts, bins = np.histogram(ages, bins=range(0, 118))

    # Inverse-frequency weights
    weights_per_age = 1.0 / (counts + 1e-5)

    # Assign a weight to each sample
    bin_indices = np.digitize(ages, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(weights_per_age) - 1)

    weights = torch.from_numpy(weights_per_age[bin_indices]).double()

    # Sample with replacement to rebalance rare ages
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler

# ------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------
def train():
    os.makedirs("results_age_balanced", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("Loading UTKFace dataset...")
    dataset = UTKFaceDataset(
        root_dir="../data/UTKFace",
        transform=transform,
    )

    # Train / test split (90% / 10%)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_test = n_total - n_train

    train_set, test_set = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Dataset split: {len(train_set)} train (balanced) | {len(test_set)} test")

    # Balanced sampler for training only
    sampler = make_balanced_sampler(train_set.indices, dataset)

    loader_kwargs = (
        dict(num_workers=4, pin_memory=True, persistent_workers=True)
        if DEVICE.type == "cuda"
        else {}
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    # Model
    model = CCVAE_Age(
        img_channels=3,
        z_c_dim=Z_C_DIM,
        z_not_c_dim=Z_NOT_C_DIM,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    print(f"Starting balanced regression training on {DEVICE}")

    # Fixed batch for reconstruction monitoring
    fixed_test_batch = next(iter(test_loader))[0].to(DEVICE)[:8]

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        total_mae = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            recon, mu, logvar, y_pred, p_mu, p_logvar = model(x, y)

            loss, stats = loss_regression_paper(
                recon, x,
                mu, logvar,
                y_pred, y,
                p_mu, p_logvar,
                gamma=GAMMA,
                beta=BETA,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                mae = torch.abs(y_pred - y).mean().item() * 100.0
                total_mae += mae

            total_batches += 1

            if total_batches % 100 == 0:
                print(
                    f"  Batch {total_batches:4d} | "
                    f"Loss: {loss.item():.1f} | "
                    f"MAE: {mae:.1f} years"
                )

        avg_loss = total_loss / total_batches
        avg_mae = total_mae / total_batches
        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"{duration:.0f}s | "
            f"Loss: {avg_loss:.1f} | "
            f"Train MAE: {avg_mae:.2f} years"
        )

        scheduler.step(avg_mae)

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            print("  Generating visualizations...")

            with torch.no_grad():
                # Reconstruction
                recon_fixed, *_ = model(fixed_test_batch)
                recon_grid = torch.cat([fixed_test_batch, recon_fixed])
                save_image(
                    recon_grid.cpu(),
                    f"results_age_balanced/recon_epoch_{epoch+1}.png",
                    nrow=8,
                )

                # Aging strip
                test_img = next(iter(test_loader))[0].to(DEVICE)[0:1]
                _, mu_enc, *_ = model(test_img)

                z_not_c = mu_enc[:, Z_C_DIM:]

                ages = torch.linspace(0.1, 0.9, 9).unsqueeze(1).to(DEVICE)
                h_prior = model.prior_net(ages)
                z_c_targets = model.prior_mu(h_prior)

                z_not_c_rep = z_not_c.repeat(9, 1)
                z_full = torch.cat([z_c_targets, z_not_c_rep], dim=1)

                dec_in = model.decoder_input(z_full).view(-1, 512, 4, 4)
                generated = model.decoder_conv(dec_in)

                aging_strip = torch.cat([test_img, generated])
                save_image(
                    aging_strip.cpu(),
                    f"results_age_balanced/aging_epoch_{epoch+1}.png",
                    nrow=10,
                )

            torch.save(model.state_dict(), "ccvae_age_balanced.pth")
            print("  Images and model saved.")

if __name__ == "__main__":
    train()
