import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np

# Tes fichiers
from model import CCVAE
from dataset import CartoonDataset

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_PATH = "./cartoonset10k/cartoonset10k"
MODEL_PATH = "ccvae_cartoon.pth"

# C'est ICI que tu choisis quel attribut visualiser (0 à 17 généralement)
# Essaie de changer cette valeur si l'image ne change pas assez !
ATTRIBUTE_INDEX_TO_VARY = 1 

def visualize_traversal():
    # 1. Charger une image du dataset pour auto-détecter les dimensions
    # (Exactement comme dans le train pour être cohérent)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # On charge juste quelques images pour lire la config
    dataset = CartoonDataset(root_dir=IMG_PATH, transform=transform, num_images=10)
    
    test_img, test_label = dataset[0]
    num_attributes = len(test_label)
    print(f"--> Configuration détectée : {num_attributes} attributs (Z_C_DIM)")

    # 2. Initialiser le modèle avec ces dimensions
    model = CCVAE(
        img_channels=3, 
        z_c_dim=num_attributes, 
        z_not_c_dim=32, 
        num_labels=num_attributes
    ).to(DEVICE)

    # 3. Charger les poids entraînés
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"Erreur de chargement : {e}")
        print("Vérifie que 'ccvae_cartoon.pth' existe et que les dimensions correspondent.")
        return

    model.eval()
    
    # 4. Préparer l'image de base (celle dont on garde le style)
    base_img = test_img.unsqueeze(0).to(DEVICE) # (1, 3, 64, 64)
    
    # 5. Encodage pour extraire le style (z_not_c)
    with torch.no_grad():
        h = model.encoder_conv(base_img)
        mu = model.fc_mu(h)
        
        # On garde la partie "Style" (z_not_c) FIXE
        # z_c est au début, z_not_c est à la fin
        z_not_c_fixed = mu[:, num_attributes:] 
        
        # On récupère aussi le z_c original pour référence
        z_c_original = mu[:, :num_attributes]
        
        # 6. Créer la traversée (Latent Traversal)
        # On va faire varier UN SEUL neurone de z_c (celui choisi plus haut)
        steps = 10
        # On va de -3 à +3 écarts-types
        values = torch.linspace(-3, 3, steps).to(DEVICE)
        
        generated_images = []
        
        for val in values:
            # On part du z_c original de l'image
            z_c_new = z_c_original.clone()
            
            # On écrase juste la valeur de l'attribut choisi
            z_c_new[0, ATTRIBUTE_INDEX_TO_VARY] = val
            
            # Concaténation
            z_combined = torch.cat([z_c_new, z_not_c_fixed], dim=1)
            
            # Décodage
            dec_input = model.decoder_input(z_combined).view(-1, 64, 4, 4)
            img_recon = model.decoder_conv(dec_input)
            
            generated_images.append(img_recon.cpu().squeeze(0))
            
    # 7. Affichage
    grid = make_grid(generated_images, nrow=steps, padding=2, pad_value=1)
    
    plt.figure(figsize=(15, 4))
    # permute pour passer de (C, H, W) à (H, W, C) pour matplotlib
    plt.imshow(grid.permute(1, 2, 0)) 
    plt.axis('off')
    plt.title(f"Variation de l'attribut n°{ATTRIBUTE_INDEX_TO_VARY} (Style conservé)")
    plt.show()
    print(f"Figure générée. Si l'image change peu, essaie un autre ATTRIBUTE_INDEX_TO_VARY.")

if __name__ == "__main__":
    visualize_traversal()