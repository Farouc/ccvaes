import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image # Pour visualiser
import os

from model import CCVAE
from dataset import CartoonHairColorDataset
# On suppose que tes loss sont dans loss_paper.py
from loss import ccvae_loss_supervised_paper, ccvae_loss_unsupervised_paper

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30
LABELED_RATIO = 0.20   
K_SAMPLES = 10          # Réduit à 5 pour éviter le Out-Of-Memory si GPU modeste
ALPHA = 1.0            # Poids de la loss non-supervisée (optionnel)

def split_supervised(dataset, labeled_ratio):
    n_total = len(dataset)
    n_labeled = int(n_total * labeled_ratio)
    n_unlabeled = n_total - n_labeled
    # Fixer le seed pour la reproductibilité est crucial ici
    return random_split(dataset, [n_labeled, n_unlabeled], 
                        generator=torch.Generator().manual_seed(42))

def train():
    os.makedirs("results", exist_ok=True) # Dossier pour sauver les images

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    print("Chargement du dataset...")
    # Assure-toi que le chemin est bon
    dataset = CartoonHairColorDataset(root_dir="./cartoonset10k/cartoonset10k", transform=transform)
    num_classes = dataset.num_classes

    labeled_set, unlabeled_set = split_supervised(dataset, LABELED_RATIO)
    
    # NOTE: On met drop_last=True pour éviter les bugs de batch size variable à la fin
    labeled_loader = DataLoader(labeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"--> Labeled: {len(labeled_set)} | Unlabeled: {len(unlabeled_set)}")

    model = CCVAE(
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=64, # 32 ou 64 selon ton model.py
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Démarrage entraînement...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        total_batches = 0

        # CORRECTION 1: On crée un itérateur "infini" pour le petit dataset
        labeled_iter = iter(labeled_loader)

        # CORRECTION 1: On boucle sur le GRAND dataset (unlabeled)
        for x_unlabeled, _ in unlabeled_loader:
            x_unlabeled = x_unlabeled.to(DEVICE)

            # Récupération du batch labeled (avec cycle infini)
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(DEVICE)
            y_labeled = y_labeled.to(DEVICE)

            optimizer.zero_grad()

            # --- A. Supervised Step ---
            loss_sup, stats_sup = ccvae_loss_supervised_paper(
                model, x_labeled, y_labeled, K=K_SAMPLES,recon_type="bce"
            )

            # --- B. Unsupervised Step ---
            loss_unsup, stats_unsup = ccvae_loss_unsupervised_paper(
                model, x_unlabeled, K=K_SAMPLES,recon_type="bce"
            )

            # Somme des pertes
            loss = loss_sup + ALPHA * loss_unsup

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            # CORRECTION 2 : Calcul de l'accuracy sur le batch supervisé
            # On réutilise les logits calculés implicitement ? 
            # Non, il faut refaire un petit forward rapide ou modifier la loss pour retourner les logits.
            # Pour faire simple ici, on refait un pass inference rapide sur la partie classification.
            with torch.no_grad():
                # On encode juste pour avoir z_c
                # Note: c'est coûteux de refaire tout le forward, idéalement
                # modifie ccvae_loss_supervised pour qu'elle te renvoie les preds.
                # Ici on fait une approximation rapide :
                h = model.encoder_conv(x_labeled)
                mu = model.fc_mu(h)
                z_c = mu[:, :model.z_c_dim] # On utilise la moyenne, pas le sample
                logits = model.classifier(z_c)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_labeled).float().mean()
                total_acc += acc.item()

            if total_batches % 50 == 0:
                print(f"   Batch {total_batches} | Loss: {loss.item():.2f} (Sup: {loss_sup.item():.2f}) | Acc: {acc.item():.2%}")

        # Fin de l'époque
        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        print(f"[Epoch {epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2%}")

        # CORRECTION 3 : Sauvegarder des reconstructions pour vérifier visuellement
        with torch.no_grad():
            # On prend 8 images du batch non supervisé
            test_x = x_unlabeled[:8]
            recon_x, _, _, _, _, _ = model(test_x)
            comparison = torch.cat([test_x, recon_x])
            save_image(comparison.cpu(), f"results/recon_epoch_{epoch+1}.png", nrow=8)

        torch.save(model.state_dict(), "ccvae_haircolor.pth")

if __name__ == "__main__":
    train()