import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- IMPORTS FROM YOUR PROJECT STRUCTURE ---
# We add the current directory to sys.path to ensure we can import 'src'
sys.path.append(os.getcwd())

from src.dataset.utk_faces import UTKFaceDataset
# from src.dataset.cartoonset import CartoonSetDataset # Uncomment if needed
from src.models.ccvae import CCVAE
from src.loss import get_loss_function

# --- HELPER: SAVE PLOTS ---
def save_reconstruction_plot(original, reconstructed, epoch, save_dir, n_images=8):
    """
    Saves a grid of images: Top row = Original, Bottom row = Reconstruction.
    """
    model_output_device = original.device
    orig = original[:n_images].detach().cpu()
    recon = reconstructed[:n_images].detach().cpu()
    
    # Create a figure
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
    
    for i in range(min(n_images, len(orig))):
        # Original Image: (C, H, W) -> (H, W, C)
        img_orig = orig[i].permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1) # Ensure valid range
        
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        # Reconstructed Image
        img_recon = recon[i].permute(1, 2, 0).numpy()
        img_recon = np.clip(img_recon, 0, 1)
        
        axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    filename = os.path.join(save_dir, f"recon_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved reconstruction figure: {filename}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- MAIN TRAINING LOOP ---
def train():
    parser = argparse.ArgumentParser(description="CCVAE Training Script")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    
    # CLI Overrides
    parser.add_argument('--model_size', type=str, help='Override model size (small, medium, large)')
    parser.add_argument('--loss', type=str, help='Override loss type (paper, ours, contrastive)')
    parser.add_argument('--task', type=str, help='Override task type (regression, classification_mono, etc)')
    
    args = parser.parse_args()
    
    # 1. Load Configuration
    cfg = load_config(args.config)
    
    # 2. Apply Overrides (CLI takes precedence over config.yaml)
    current_size = args.model_size if args.model_size else cfg['model_size']
    current_loss = args.loss if args.loss else cfg['loss_type']
    current_task = args.task if args.task else cfg['task_type']
    
    # 3. Resolve Output Dimension based on Task
    # This fixes the mismatch error you saw earlier
    if current_task == 'regression':
        current_output_dim = 1
    elif current_task == 'classification_mono':
        current_output_dim = 2
    else:
        # For multi-class, trust the config or default to something specific
        current_output_dim = cfg.get('output_dim', 2)

    # 4. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['experiment_name']}_{current_size}_{current_loss}_{current_task}_{timestamp}"
    save_dir = os.path.join(cfg['output_dir'], run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the effective config for reproducibility
    cfg['actual_run_params'] = {
        'model_size': current_size,
        'loss_type': current_loss, 
        'task_type': current_task,
        'output_dim': current_output_dim
    }
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print(f"--- Starting Training ---")
    print(f"Model Size: {current_size}")
    print(f"Loss Type : {current_loss}")
    print(f"Task Type : {current_task} (Output Dim: {current_output_dim})")
    print(f"Saving to : {save_dir}")

    # 5. Initialize Dataset & Dataloader
    # Using the class from src/dataset/utk_faces.py
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
    ])
    
    print(f"Loading dataset from: {cfg['dataset_path']}")
    dataset = UTKFaceDataset(
        root_dir=cfg['dataset_path'], 
        transform=transform, 
        task_type=current_task
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=cfg.get('num_workers', 0)
    )

    # 6. Initialize Model
    # Using the class from src/models/ccvae.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = cfg['model_configs'][current_size]
    
    model = CCVAE(
        config=model_params, 
        output_dim=current_output_dim, 
        task_type=current_task
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # 7. Initialize Loss
    # Using the function from src/models/loss.py
    loss_fn = get_loss_function(current_loss, current_task)

    # 8. Training Loop
    epochs = cfg['training']['epochs']
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Expecting model to return: reconstruction, prediction, features
            recon, pred, features = model(images)
            
            # Calculate Loss
            # Loss fn expects: recon, x, pred, target, features
            total_loss, l_recon, l_task = loss_fn(recon, images, pred, labels, features)
            
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {total_loss.item():.4f} (Recon: {l_recon.item():.4f}, Task: {l_task.item():.4f})")

        # --- SAVING & VISUALIZATION ---
        if epoch % cfg['training']['save_interval'] == 0 or epoch == epochs:
            # Save Weights
            weight_name = f"model_ep{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, weight_name))
            print(f"Saved weights: {weight_name}")
            
            # Save Reconstruction Figure
            save_reconstruction_plot(images, recon, epoch, save_dir)

if __name__ == "__main__":
    train()