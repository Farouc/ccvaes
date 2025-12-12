import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
import os
import sys
import time
import numpy as np

# Add root project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ccvae import CCVAE
from src.dataset.utk_faces import UTKFaceDataset
from src.loss import ccvae_loss_simple_analytic

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 200
IMG_SIZE = 128

# Dimensions
Z_C_DIM = 16    # Age
Z_NOT_C_DIM = 64
GAMMA = 100.0   # Regression weight
BETA = 0.001    # KL weight

def make_balanced_sampler(dataset_indices, dataset_full):
    print("Calculating weights for balanced sampling...")
    subset_ages = []
    for idx in dataset_indices:
        img_name = dataset_full.image_files[idx]
        try:
            age = int(img_name.split('_')[0])
            subset_ages.append(age)
        except:
            subset_ages.append(25)
            
    subset_ages = np.array(subset_ages)
    counts, bins = np.histogram(subset_ages, bins=range(0, 118))
    weights_per_age = 1.0 / (counts + 1e-5)
    
    indices_bins = np.digitize(subset_ages, bins[:-1]) - 1
    indices_bins = np.clip(indices_bins, 0, len(weights_per_age) - 1)
    
    weights = weights_per_age[indices_bins]
    weights = torch.from_numpy(weights).double()
    
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def train():
    os.makedirs("results/regression", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("Loading UTKFace dataset...")
    dataset = UTKFaceDataset(root_dir="data/UTKFace", transform=transform)
    
    # 90% Train / 10% Test
    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_test = n_total - n_train
    
    train_set, test_set = random_split(dataset, [n_train, n_test], 
                                       generator=torch.Generator().manual_seed(42))
    
    print(f"--> Data: {len(train_set)} Train | {len(test_set)} Test")

    sampler = make_balanced_sampler(train_set.indices, dataset)
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, drop_last=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)

    # Model for Regression
    model = CCVAE(
        img_channels=3,
        img_size=IMG_SIZE,
        z_c_dims=[Z_C_DIM],
        z_not_c_dim=Z_NOT_C_DIM,
        task_types=['regression'],
        num_classes_or_dim=[1],
        dropout_p=0.0
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print(f"Starting Regression Training on {DEVICE}...")
    fixed_test_batch = next(iter(test_loader))[0].to(DEVICE)[:8]

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_mae = 0
        total_batches = 0
        
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE) # (B, 1)

            optimizer.zero_grad()

            # Using Analytic Loss for stability in regression
            # Note: The model forward is called inside this loss function wrapper
            loss, stats = ccvae_loss_simple_analytic(
                model, x, y, 
                beta=BETA, 
                gamma=GAMMA, 
                recon_type="mse"
            )

            # We need to manually backprop the returned scalar loss
            # (In my previous code for loss.py, it returned the scalar loss, so we just call backward)
            # However, `ccvae_loss_simple_analytic` calls model forward internally.
            # To avoid double forward pass, check if you want to optimize this structure.
            # For now, let's stick to the function logic.
            
            # Re-implementation note: Since `ccvae_loss_simple_analytic` does the forward pass,
            # we need to ensure the loss tensor is connected to the graph. 
            # The function I wrote returns `avg_loss` which is a tensor. 
            # So:
            loss_tensor = torch.tensor(loss, requires_grad=True) if not isinstance(loss, torch.Tensor) else loss
            loss_tensor.backward()
            optimizer.step()

            # Calculate MAE for monitoring (y is normalized 0-1, so *100 = years)
            # We need predictions. Since loss func doesn't return preds, we can grab them
            # via a quick re-forward or by modifying the loss signature.
            # Efficient fix: Modify `loss.py` to return preds, OR just do a no_grad forward here for metrics?
            # Actually, `ccvae_loss_simple_analytic` returns `stats` dict but not preds.
            # For simplicity in this script, let's just trust the loss descent for now.
            
            total_loss += loss.item()
            total_batches += 1

            if total_batches % 100 == 0:
                print(f"   Batch {total_batches} | Loss: {loss.item():.4f}")

        # Metrics & Scheduler
        avg_loss = total_loss / total_batches
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | {duration:.0f}s | Loss: {avg_loss:.4f}")
        
        # Test Set MAE Calculation
        model.eval()
        mae_sum = 0
        batches_test = 0
        with torch.no_grad():
            for xt, yt in test_loader:
                xt, yt = xt.to(DEVICE), yt.to(DEVICE)
                _, _, _, preds, _, _ = model(xt)
                y_pred = preds[0]
                mae_sum += torch.abs(y_pred - yt).mean().item() * 100
                batches_test += 1
        
        avg_mae_test = mae_sum / batches_test
        print(f"   Test MAE: {avg_mae_test:.2f} years")
        scheduler.step(avg_mae_test)

        # Visualizations
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                # Recon
                recon_fixed, _, _, _, _, _ = model(fixed_test_batch)
                comparison = torch.cat([fixed_test_batch, recon_fixed])
                save_image(comparison.cpu(), f"results/regression/recon_epoch_{epoch+1}.png", nrow=8)
                
        torch.save(model.state_dict(), "results/regression/ccvae_utk.pth")

if __name__ == "__main__":
    train()