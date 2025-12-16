# ============================================================
#   UTKFace – Age Regression Benchmarks (Clean & Fixed)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import warnings
from pathlib import Path

from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

# ------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataset import UTKFaceDataset
except ImportError:
    print("FATAL ERROR: Could not import UTKFaceDataset.")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../data/UTKFace"
TARGET_LABEL = "Age"

IMAGE_SIZE = 64
AGE_MAX = 100.0          # used only for de-normalization

N_SPLITS = 5
N_CNN_EPOCHS = 10
PCA_COMPONENTS = 256
SAMPLE_LIMIT = 5000

# ------------------------------------------------------------
# 2. CNN REGRESSION MODEL
# ------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    CNN for normalized age regression in [0, 1].
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_size = 1024
        self.head = nn.Linear(self.fc_size, 1)

    def forward(self, x):
        h = self.features(x)
        # constrain prediction to [0,1] since target is normalized
        return torch.sigmoid(self.head(h))

# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def extract_features_and_labels(dataset, sample_limit, pca_dim=None):
    """
    Extract flattened pixels and normalized age labels.
    """
    print(f"\n[Data] Extracting {sample_limit} samples for classical models")

    indices = np.random.RandomState(42).permutation(len(dataset))[:sample_limit]
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=256, shuffle=False)

    X, y = [], []

    for imgs, labels in tqdm(loader, desc="  > Flattening"):
        X.append(imgs.view(imgs.size(0), -1).numpy())
        y.extend(labels.numpy().astype(np.float32))

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    if pca_dim is not None and pca_dim < X.shape[1]:
        print(f"[Data] PCA: {X.shape[1]} → {pca_dim}")
        X = PCA(n_components=pca_dim, random_state=42).fit_transform(X)

    return X, y


def evaluate_classical_model(X, y, name, model_class, **kwargs):
    """
    K-Fold CV for classical regressors.
    Metric: MAE in years.
    """
    print(f"\n--- {name} ---")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    maes = []

    for tr, te in tqdm(kf.split(X), total=N_SPLITS, leave=False):
        model = model_class(**kwargs)
        model.fit(X[tr], y[tr])

        pred = model.predict(X[te])
        mae = mean_absolute_error(y[te] * AGE_MAX, pred * AGE_MAX)
        maes.append(mae)

    avg = float(np.mean(maes))
    print(f"  > MAE: {avg:.2f} years")
    return [avg]


def train_and_evaluate_cnn(dataset, n_epochs, split_ratio=0.8):
    """
    Single split CNN regression baseline.
    Metric: MAE in years.
    """
    print(f"\n--- CNN Regression ({n_epochs} epochs) ---")

    n_total = len(dataset)
    n_train = int(split_ratio * n_total)

    train_set, test_set = random_split(
        dataset,
        [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- Training ----
    for _ in tqdm(range(n_epochs), desc="  > Training"):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # ---- Evaluation ----
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE).float()
            pred = model(x).cpu().numpy().ravel()

            y_pred.extend(pred)
            y_true.extend(y.numpy())

    y_true = np.array(y_true) * AGE_MAX
    y_pred = np.clip(np.array(y_pred), 0.0, 1.0) * AGE_MAX

    mae = mean_absolute_error(y_true, y_pred)
    print(f"  > MAE: {mae:.2f} years")

    return [mae]

# ------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------

def main():
    print(f"\n=== UTKFace Age Regression ({DEVICE}) ===")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3)
    ])

    dataset = UTKFaceDataset(DATA_DIR, transform=transform)

    X, y = extract_features_and_labels(
        dataset,
        sample_limit=SAMPLE_LIMIT,
        pca_dim=PCA_COMPONENTS
    )

    results = {}

    results["LinReg*"] = evaluate_classical_model(
        X, y, "Linear Regression", LinearRegression, n_jobs=-1
    )

    results["RanFor*"] = evaluate_classical_model(
        X, y, "Random Forest", RandomForestRegressor,
        n_estimators=50, random_state=42, n_jobs=-1
    )

    results["KNN"] = evaluate_classical_model(
        X, y, "KNN", KNeighborsRegressor,
        n_neighbors=5, n_jobs=-1
    )

    results["SVR"] = evaluate_classical_model(
        X, y, "SVR (Linear)", SVR,
        kernel="linear", C=1.0, max_iter=1000
    )

    results["CNN"] = train_and_evaluate_cnn(dataset, N_CNN_EPOCHS)

    # ---- Summary ----
    print("\n" + "=" * 50)
    print(" UTKFACE AGE REGRESSION RESULTS ".center(50))
    print(" Metric: MAE (years) ".center(50))
    print("=" * 50)
    print(f"| {'Model':<20} | {'MAE':<15} |")
    print(f"|{'-'*22}|{'-'*17}|")

    for k, v in results.items():
        print(f"| {k:<20} | {v[0]:<15.2f} |")

    print("=" * 50)


if __name__ == "__main__":
    main()
