import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

# Scikit-Learn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from model import CCVAE
from dataset import CartoonMultiLabelDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data/cartoonset10k/cartoonset10k"
CCVAE_WEIGHTS =  "./model_weights/ccvae_multilabel_base.pth"

TARGET_ATTRIBUTES = ["hair_color", "face_color"]
IMAGE_SIZE = 64
BATCH_SIZE = 128

# Benchmark Settings
N_SPLITS = 5          # 5-Fold Cross Validation
N_CNN_EPOCHS = 10     # Epochs for baseline CNN
PCA_COMPONENTS = 256  # Reduce features for classical models
SAMPLE_LIMIT = 300  # Limit classical models to 5k images for speed

# Silence Scikit-Learn warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. BASELINE CNN MODEL
# ============================================================
class SimpleCNN(nn.Module):
    """A simple baseline CNN for 64x64 images."""
    def __init__(self, num_classes_list):
        super(SimpleCNN, self).__init__()
        # Encoder
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_size = 1024 
        
        # Heads
        self.head_hair = nn.Linear(self.fc_size, num_classes_list[0])
        self.head_face = nn.Linear(self.fc_size, num_classes_list[1])

    def forward(self, x):
        h = self.features(x)
        return self.head_hair(h), self.head_face(h)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def extract_features_and_labels(dataset, target_attributes, sample_limit, pca_dim=None):
    """Extracts flattened pixel features for classical models (Logistic Reg, SVM, etc)."""
    print(f"\n[Data] Extracting features from {sample_limit} images for classical baselines...")
    
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    subset_indices = indices[:sample_limit]
    
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=256, num_workers=0, shuffle=False)
    
    X_flat = []
    y_multilabel_list = [[] for _ in target_attributes]

    for img, labels in tqdm(loader, desc=" > Flattening"):
        X_flat.append(img.view(img.size(0), -1).numpy())
        for i, col in enumerate(labels.T):
            y_multilabel_list[i].extend(col.tolist())
            
    X_flat = np.concatenate(X_flat, axis=0)
    Y_multi = np.stack([np.array(y) for y in y_multilabel_list], axis=1)

    if pca_dim and pca_dim < X_flat.shape[1]:
        print(f"[Data] Applying PCA ({X_flat.shape[1]} -> {pca_dim} features)...")
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced = pca.fit_transform(X_flat)
        return X_reduced, Y_multi

    return X_flat, Y_multi


def evaluate_classical_model(X, Y, model_name, base_estimator, **kwargs):
    """Runs 5-Fold CV for Scikit-Learn models wrapped in MultiOutputClassifier."""
    print(f"\n--- Benchmarking Multi-Output {model_name} ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    base_clf = base_estimator(**kwargs)
    clf = MultiOutputClassifier(base_clf)

    fold_scores_hair = []
    fold_scores_face = []
    
    # We split based on the first label just to get indices, but validation is done on both
    pbar = tqdm(skf.split(X, Y[:, 0]), total=N_SPLITS, desc=f" > CV Progress", leave=False)
    
    for train_idx, test_idx in pbar:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        acc_hair = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
        acc_face = accuracy_score(Y_test[:, 1], Y_pred[:, 1])
        
        fold_scores_hair.append(acc_hair)
        fold_scores_face.append(acc_face)
        
    avg_acc_hair = np.mean(fold_scores_hair)
    avg_acc_face = np.mean(fold_scores_face)
    
    print(f" > hair_color | Avg Accuracy: {avg_acc_hair:.2%}")
    print(f" > face_color | Avg Accuracy: {avg_acc_face:.2%}")
    
    return [avg_acc_hair, avg_acc_face]


def evaluate_cnn(dataset, num_classes_list):
    """Trains and evaluates the SimpleCNN baseline (freshly trained)."""
    print(f"\n--- Benchmarking CNN (Full Dataset, {N_CNN_EPOCHS} Epochs) ---")
    
    n_train = int(len(dataset) * 0.8)
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    model = SimpleCNN(num_classes_list).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in tqdm(range(N_CNN_EPOCHS), desc=" > Training CNN"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out_hair, out_face = model(x)
            loss = F.cross_entropy(out_hair, y[:, 0]) + F.cross_entropy(out_face, y[:, 1])
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = [0, 0]
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out_hair, out_face = model(x)
            
            pred_hair = out_hair.argmax(dim=1)
            pred_face = out_face.argmax(dim=1)
            
            correct[0] += (pred_hair == y[:, 0]).sum().item()
            correct[1] += (pred_face == y[:, 1]).sum().item()
            total += y.size(0)
            
    acc_hair = correct[0] / total
    acc_face = correct[1] / total
    
    print(f" > hair_color | Accuracy: {acc_hair:.2%}")
    print(f" > face_color | Accuracy: {acc_face:.2%}")
    
    return [acc_hair, acc_face]


def evaluate_ccvae_pretrained(dataset, model_path, device, n_splits=5):
    """
    Evaluates a PRE-TRAINED CCVAE using K-Fold Cross Validation.
    Since the model is frozen, this checks if the model's accuracy is consistent across data splits.
    """
    print(f"\n--- Benchmarking Pre-trained CCVAE (Inference Only) ---")
    
    # Define CCVAE Config (Must match training config)
    Z_C_DIMS = [16, 16] 
    Z_NOT_C_DIM = 32
    
    model = CCVAE(
        img_channels=3,
        z_c_dims=Z_C_DIMS,
        z_not_c_dim=Z_NOT_C_DIM,
        num_classes_list=[10, 11] # [Hair, Face]
    ).to(device)

    
    state = torch.load(model_path, map_location=DEVICE)
    new_state = {}
    for k, v in state.items():
        k_new = k
        k_new = k_new.replace("priors_embedding", "prior_embeddings")
        k_new = k_new.replace("priors_mu", "prior_mu")
        k_new = k_new.replace("priors_logvar", "prior_logvar")
        new_state[k_new] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    model.eval()
    

    # Collect all labels for Stratified Split
    print(" [Data] collecting labels for stratification...")
    labels_all = []
    # Using a simple loader to grab labels without loading all images to RAM
    temp_loader = DataLoader(dataset, batch_size=256, num_workers=0)
    for _, y in temp_loader:
        labels_all.append(y)
    labels_all = torch.cat(labels_all, dim=0).numpy() # Shape (N, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores_hair = []
    fold_scores_face = []

    pbar = tqdm(skf.split(np.zeros(len(labels_all)), labels_all[:, 0]), 
                total=n_splits, desc=" > CV Progress (CCVAE)", leave=False)

    for _, test_idx in pbar:
        test_subset = Subset(dataset, test_idx)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        correct_hair = 0
        correct_face = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                
                # --- CCVAE Forward Pass ---
                h = model.encoder_conv(x)
                mu = model.fc_mu(h)
                
                # Split Latents: First 16=Hair, Next 16=Face
                z_c_hair = mu[:, 0:16]
                z_c_face = mu[:, 16:32]
                
                # Classifier Heads
                logits_hair = model.classifiers[0](z_c_hair)
                logits_face = model.classifiers[1](z_c_face)
                
                pred_hair = logits_hair.argmax(dim=1)
                pred_face = logits_face.argmax(dim=1)
                
                correct_hair += (pred_hair == y[:, 0]).sum().item()
                correct_face += (pred_face == y[:, 1]).sum().item()
                total += y.size(0)
        
        fold_scores_hair.append(correct_hair / total)
        fold_scores_face.append(correct_face / total)

    avg_acc_hair = np.mean(fold_scores_hair)
    avg_acc_face = np.mean(fold_scores_face)
    
    print(f" > hair_color | Avg Accuracy: {avg_acc_hair:.2%}")
    print(f" > face_color | Avg Accuracy: {avg_acc_face:.2%}")
    
    return [avg_acc_hair, avg_acc_face]


# ============================================================
# 3. MAIN EXECUTION
# ============================================================
def main():
    print(f"=== Starting Multi-Label Benchmark on {DEVICE} ===")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = CartoonMultiLabelDataset(
        root_dir=DATA_DIR, 
        target_attributes=TARGET_ATTRIBUTES, 
        transform=transform
    )

    final_results = {}

    # 2. Classical Models (Reduced Features)
    X_reduced, Y_multi = extract_features_and_labels(
        dataset, TARGET_ATTRIBUTES, 
        sample_limit=SAMPLE_LIMIT, 
        pca_dim=PCA_COMPONENTS
    )
    # CCVAE (Pre-trained Evaluation)
    final_results['CCVAE (Frozen)'] = evaluate_ccvae_pretrained(
        dataset, 
        model_path=CCVAE_WEIGHTS, 
        device=DEVICE, 
        n_splits=N_SPLITS
    )

    final_results['LoReg*'] = evaluate_classical_model(
        X_reduced, Y_multi, "Logistic Regression", LogisticRegression,
        solver='lbfgs', max_iter=500, class_weight='balanced', n_jobs=-1
    )

    final_results['RanFor*'] = evaluate_classical_model(
        X_reduced, Y_multi, "Random Forest", RandomForestClassifier,
        n_estimators=50, class_weight='balanced', n_jobs=-1
    )

    final_results['KNN'] = evaluate_classical_model(
        X_reduced, Y_multi, "K-Nearest Neighbors", KNeighborsClassifier,
        n_neighbors=5, n_jobs=-1
    )

    final_results['SVM'] = evaluate_classical_model(
        X_reduced, Y_multi, "Support Vector Machine", SVC,
        kernel='linear', max_iter=1000, class_weight='balanced'
    )

    # 3. Baseline CNN (Full Training)
    final_results['CNN'] = evaluate_cnn(dataset, dataset.num_classes_list)

    

    # 5. Final Table
    print("\n" + "="*70)
    print(f"{' ':^70}")
    print(f"{'FINAL BENCHMARK RESULTS':^70}")
    print(f"{' (Classical models use PCA features / CNN & CCVAE use raw images)':^70}")
    print("="*70)
    
    print(f"| {'Model':<20} | {'Hair Color':<15} | {'Face Color':<15} |")
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|")
    
    for model_name, scores in final_results.items():
        hair_acc = scores[0]
        face_acc = scores[1]
        print(f"| {model_name:<20} | {hair_acc:.2%}         | {face_acc:.2%}         |")
        
    print("="*70)
    print("\nDone.")

if __name__ == "__main__":
    main()