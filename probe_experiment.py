import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CCVAE_Age
from dataset import UTKFaceDataset
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt


# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./ccvae_age_balanced.pth"
DATA_DIR = "./UTKFace"

Z_C_DIM = 16
Z_NOT_C_DIM = 64
BATCH_SIZE = 32

# --- DATASET ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = CCVAE_Age(
    img_channels=3,
    z_c_dim=Z_C_DIM,
    z_not_c_dim=Z_NOT_C_DIM
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Modèle chargé et prêt !")

# --- EXTRACT Z_not_c FEATURES AND AGE TARGETS ---
all_z_not_c = []
all_ages = []

num_images = 0
max_images = 20000

with torch.no_grad():
    for imgs, ages_norm in dataloader:
        # Move to device
        imgs = imgs.to(DEVICE)
        ages = ages_norm.squeeze() * 100
        
        # Forward pass
        recon, mu, _, pred_age_norm, _, _ = model(imgs)
        z_not_c_batch = mu[:, Z_C_DIM:]
        
        all_z_not_c.append(z_not_c_batch.cpu().numpy())
        all_ages.append(ages.cpu().numpy())
        
        # Count images
        num_images += imgs.size(0)
        if num_images >= max_images:
            break

# Stack everything
X = np.vstack(all_z_not_c)  # shape: [num_samples, Z_NOT_C_DIM]
y = y = np.concatenate(all_ages)     # shape: [num_samples]

print(f"Dataset created: X={X.shape}, y={y.shape}")

# --- SPLIT TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- TRAIN REGRESSOR ---
regressor = LinearRegression()
# Optional: use Ridge if you want some regularization
# regressor = Ridge(alpha=1.0)

regressor.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("=== Leakage Probe Results ===")
print(f"R² score: {r2:.4f}")
print(f"MAE: {mae:.2f} years")
print(f"RMSE: {rmse:.2f} years")


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Age")
plt.ylabel("Predicted Age from z_not_c")
plt.title("Leakage Probe: Age from z_not_c")
plt.plot([0, 100], [0, 100], 'r--')  # perfect prediction line
plt.show()