import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier # New Import for Multi-Label

# --- 1. SETUP & CONFIGURATION ---

# Silence Scikit-Learn warnings (e.g. convergence warnings, deprecation) for clean output
warnings.filterwarnings("ignore")

# Assuming tqdm is available, if not, it should be installed via pip
from tqdm import tqdm 

# Path Management: Add project root to path to import from classification_mono
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CartoonMultiLabelDataset

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data/cartoonset10k/cartoonset10k"
TARGET_ATTRIBUTES = ["hair_color", "face_color"]
IMAGE_SIZE = 64

# Benchmark Settings
N_SPLITS = 5          # 5-Fold Cross Validation
N_CNN_EPOCHS = 10     # Epochs for CNN
PCA_COMPONENTS = 256  # Reduce features from 12,288 -> 256 for classical models
SAMPLE_LIMIT = 5000   # Limit classical models to 5k images for speed

# ------------------------------------------------------------
# 2. CNN MODEL DEFINITION
# ------------------------------------------------------------
class SimpleCNN(nn.Module):
    """A simple baseline CNN for 64x64 images."""
    def __init__(self, num_classes_list):
        super(SimpleCNN, self).__init__()
        # Encoder (Matches VAE structure roughly)
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

# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------
def extract_features_and_labels(dataset, target_attributes, sample_limit, pca_dim=None):
    """
    Extracts raw flattened pixels and combines labels into a multi-output matrix Y.
    """
    print(f"\n[Data] Extracting features from {sample_limit} images...")
    
    # 1. Random Subset
    indices = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    subset_indices = indices[:sample_limit]
    
    from torch.utils.data import Subset
    subset = Subset(dataset, subset_indices)
    
    # 2. Extraction Loop
    loader = DataLoader(subset, batch_size=256, num_workers=0, shuffle=False)
    X_flat = []
    y_multilabel_list = [[] for _ in target_attributes]

    for img, labels in tqdm(loader, desc="  > Flattening"):
        # Flatten (B, C, H, W) -> (B, -1)
        X_flat.append(img.view(img.size(0), -1).numpy())
        for i, col in enumerate(labels.T):
            y_multilabel_list[i].extend(col.tolist())
            
    X_flat = np.concatenate(X_flat, axis=0)
    
    # Combine y_list into a single (N_samples, N_labels) matrix Y for multi-output training
    Y_multi = np.stack([np.array(y) for y in y_multilabel_list], axis=1)

    # 3. PCA Reduction
    if pca_dim and pca_dim < X_flat.shape[1]:
        print(f"[Data] Applying PCA ({X_flat.shape[1]} -> {pca_dim} features)...")
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced = pca.fit_transform(X_flat)
        return X_reduced, Y_multi

    return X_flat, Y_multi

def evaluate_classical_model(X, Y, model_name, base_estimator, **kwargs):
    """
    Runs Cross-Validation for a MultiOutputClassifier wrapped around a base estimator.
    """
    print(f"\n--- Benchmarking Multi-Output {model_name} ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Prepare the base estimator with specified kwargs
    base_clf = base_estimator(**kwargs)
    
    # Wrap it for multi-output training
    clf = MultiOutputClassifier(base_clf)

    fold_scores_hair = []
    fold_scores_face = []
    
    # TQDM progress bar for folds
    pbar = tqdm(skf.split(X, Y[:, 0]), total=N_SPLITS, desc=f"  > CV Progress", leave=False)
    for train_idx, test_idx in pbar:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Train the MultiOutputClassifier on the combined Y_train
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        # Evaluate accuracy for each attribute (column)
        acc_hair = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
        acc_face = accuracy_score(Y_test[:, 1], Y_pred[:, 1])
        
        fold_scores_hair.append(acc_hair)
        fold_scores_face.append(acc_face)
        
    avg_acc_hair = np.mean(fold_scores_hair)
    avg_acc_face = np.mean(fold_scores_face)
    
    print(f"  > {TARGET_ATTRIBUTES[0]:<10} | Avg Accuracy: {avg_acc_hair:.2%}")
    print(f"  > {TARGET_ATTRIBUTES[1]:<10} | Avg Accuracy: {avg_acc_face:.2%}")
    
    return [avg_acc_hair, avg_acc_face]

def evaluate_cnn(dataset, num_classes_list):
    """
    Trains and evaluates the CNN baseline on the full dataset. (This is already multi-task/multi-label)
    """
    print(f"\n--- Benchmarking CNN (Full Dataset, {N_CNN_EPOCHS} Epochs) ---")
    
    # Split
    n_train = int(len(dataset) * 0.8)
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Model Setup
    model = SimpleCNN(num_classes_list).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    for epoch in tqdm(range(N_CNN_EPOCHS), desc="  > Training CNN"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            out_hair, out_face = model(x)
            loss = F.cross_entropy(out_hair, y[:, 0]) + F.cross_entropy(out_face, y[:, 1])
            
            loss.backward()
            optimizer.step()
            
    # Evaluation
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
    
    print(f"  > hair_color | Accuracy: {acc_hair:.2%}")
    print(f"  > face_color | Accuracy: {acc_face:.2%}")
    
    return [acc_hair, acc_face]

# ------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------
def main():
    print(f"=== Starting Multi-Label Classification Benchmark on {DEVICE} ===")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = CartoonMultiLabelDataset(DATA_DIR, TARGET_ATTRIBUTES, transform=transform)

    # 2. Prepare Features for Classical Models (Reduced & Multi-Output Y)
    # X_reduced is the feature matrix, Y_multi is the (N_samples, N_labels) target matrix
    X_reduced, Y_multi = extract_features_and_labels(
        dataset, TARGET_ATTRIBUTES, 
        sample_limit=SAMPLE_LIMIT, 
        pca_dim=PCA_COMPONENTS
    )

    # Dictionary to store final results
    final_results = {}

    # 3. Run Classical Benchmarks (Multi-Output Classifier)
    
    # Logistic Regression
    final_results['LoReg*'] = evaluate_classical_model(
        X_reduced, Y_multi, "Logistic Regression", LogisticRegression,
        solver='lbfgs', max_iter=500, class_weight='balanced', n_jobs=-1
    )

    # Random Forest
    final_results['RanFor*'] = evaluate_classical_model(
        X_reduced, Y_multi, "Random Forest", RandomForestClassifier,
        n_estimators=50, class_weight='balanced', n_jobs=-1
    )

    # KNN
    final_results['KNN'] = evaluate_classical_model(
        X_reduced, Y_multi, "K-Nearest Neighbors", KNeighborsClassifier,
        n_neighbors=5, n_jobs=-1
    )

    # SVM
    final_results['SVM'] = evaluate_classical_model(
        X_reduced, Y_multi, "Support Vector Machine", SVC,
        kernel='linear', max_iter=1000, class_weight='balanced'
    )

    # 4. Run CNN Benchmark (Multi-Task Learning)
    final_results['CNN'] = evaluate_cnn(dataset, dataset.num_classes_list)

    # ------------------------------------------------------------
    # 5. FINAL SUMMARY TABLE
    # ------------------------------------------------------------
    print("\n" + "="*65)
    print(f"{' ':^65}")
    print(f"{'FINAL BENCHMARK RESULTS':^65}")
    print(f"{' (Multi-Output Classifiers used for classical models)':^65}")
    print("="*65)
    
    # Table Header
    print(f"| {'Model':<20} | {'Hair Color':<15} | {'Face Color':<15} |")
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|")
    
    # Table Rows
    for model_name, scores in final_results.items():
        hair_acc = scores[0]
        face_acc = scores[1]
        print(f"| {model_name:<20} | {hair_acc:.2%}         | {face_acc:.2%}         |")
        
    print("="*65)
    print("\nDone.")

if __name__ == "__main__":
    main()