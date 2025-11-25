import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# On importe ta classe modèle
from model import CCVAE

class CCVAEInference:
    def __init__(self, model_path, num_classes=10, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # 1. Recharger l'architecture (doit matcher exactement train.py)
        # Vérifie bien que z_c_dim et z_not_c_dim sont les mêmes qu'à l'entraînement !
        self.model = CCVAE(img_channels=3, z_c_dim=16, z_not_c_dim=64, num_classes=num_classes)
        
        # 2. Charger les poids
        print(f"Chargement des poids depuis {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval() # Important pour désactiver le Dropout/BatchNorm dynamique
        
        # 3. Préparation des transforms (identique au train)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Mapping optionnel pour l'affichage (à adapter selon ton dataset)
        self.class_names = {i: f"Couleur {i}" for i in range(num_classes)}

    def preprocess(self, image_path):
        """Charge et prépare une image."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0) # Ajout dimension batch (1, 3, 64, 64)
        return img_tensor.to(self.device)

    def show(self, tensor_img, title="Image"):
        """Affiche un tenseur PyTorch."""
        img = tensor_img.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
        # Clip pour éviter les warnings si valeurs un peu hors [0,1]
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    # ==========================================================
    # USAGE 1 : CLASSIFICATION
    # ==========================================================
    def classify_image(self, image_path):
        x = self.preprocess(image_path)
        
        with torch.no_grad():
            # Encoder
            h = self.model.encoder_conv(x)
            mu = self.model.fc_mu(h)
            
            # Extraire partie Caractéristique (z_c)
            mu_c = mu[:, :self.model.z_c_dim]
            
            # Classifier
            logits = self.model.classifier(mu_c)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
            
        print(f"--> Classification : {self.class_names[pred_idx]} ({confidence:.1%} de confiance)")
        return pred_idx

    # ==========================================================
    # USAGE 2 : GÉNÉRATION CONDITIONNELLE
    # On fixe une couleur (y) et on génère un visage aléatoire
    # ==========================================================
    def generate_from_class(self, class_idx):
        target_y = torch.tensor([class_idx]).to(self.device)
        
        with torch.no_grad():
            # 1. Prédire le z_c idéal pour cette classe via le Prior
            y_onehot = F.one_hot(target_y, num_classes=self.num_classes).float()
            y_embed = self.model.y_embedding(y_onehot)
            z_c = self.model.cond_prior_mu(y_embed) # On prend la moyenne du prior
            
            # 2. Générer un style aléatoire (z_not_c) bruit gaussien
            z_not_c = torch.randn(1, self.model.z_not_c_dim).to(self.device)
            
            # 3. Combiner et Décodeur
            z = torch.cat([z_c, z_not_c], dim=1)
            
            dec_in = self.model.decoder_input(z).view(-1, 64, 4, 4)
            recon_x = self.model.decoder_conv(dec_in)
            
        self.show(recon_x, title=f"Génération : {self.class_names[class_idx]}")

    # ==========================================================
    # USAGE 3 : TRANSFERT D'ATTRIBUT (Style Swapping)
    # On prend le visage de A et on lui force la couleur de B
    # ==========================================================
    def swap_hair_color(self, img_content_path, img_color_source_path):
        """
        Garde le visage de l'image 'Content' mais applique la couleur de 'Source'.
        """
        x_content = self.preprocess(img_content_path) # Visage/Style
        x_color = self.preprocess(img_color_source_path)   # Couleur cheveux
        
        with torch.no_grad():
            # Encoder les deux images
            # Image CONTENT (on veut z_not_c)
            h1 = self.model.encoder_conv(x_content)
            mu1 = self.model.fc_mu(h1)
            z_not_c_content = mu1[:, self.model.z_c_dim:] # Partie Droite
            
            # Image COLOR SOURCE (on veut z_c)
            h2 = self.model.encoder_conv(x_color)
            mu2 = self.model.fc_mu(h2)
            z_c_color = mu2[:, :self.model.z_c_dim]       # Partie Gauche
            
            # Combiner : [Couleur de B, Style de A]
            z_new = torch.cat([z_c_color, z_not_c_content], dim=1)
            
            # Décoder
            dec_in = self.model.decoder_input(z_new).view(-1, 64, 4, 4)
            result = self.model.decoder_conv(dec_in)
            
        # Affichage comparatif
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        # Image A
        img_A = x_content.squeeze().cpu().permute(1, 2, 0).numpy()
        axs[0].imshow(img_A)
        axs[0].set_title("Visage Original (A)")
        axs[0].axis('off')

        # Image B
        img_B = x_color.squeeze().cpu().permute(1, 2, 0).numpy()
        axs[1].imshow(img_B)
        axs[1].set_title("Source Couleur (B)")
        axs[1].axis('off')

        # Résultat
        img_res = result.squeeze().cpu().permute(1, 2, 0).numpy()
        axs[2].imshow(img_res)
        axs[2].set_title("A avec cheveux de B")
        axs[2].axis('off')
        
        plt.show()

# ==========================================================
# EXEMPLE D'EXECUTION
# ==========================================================
if __name__ == "__main__":
    # Instanciation
    predictor = CCVAEInference(model_path="ccvae_haircolor.pth", num_classes=10)
    
    # Chemins vers des images de test (à changer)
    test_img_1 = "./cartoonset10k/cartoonset10k/cs10188651804540743.png"
    test_img_2 = "./cartoonset10k/cartoonset10k/cs1048486361028912.png" # Supposons une autre couleur
    
    print("--- 1. Test Classification ---")
    try:
        class_id = predictor.classify_image(test_img_1)
    except FileNotFoundError:
        print("Erreur: Image introuvable. Ajuste le chemin.")

    print("\n--- 2. Test Génération (Création pure) ---")
    # Générer une image avec la couleur classe 3
    predictor.generate_from_class(class_idx=3)

    print("\n--- 3. Test Swap (Changement de couleur) ---")
    try:
        # On garde la tête de l'image 1, on met la couleur de l'image 2
        predictor.swap_hair_color(test_img_1, test_img_2)
    except FileNotFoundError:
        print("Erreur: Images introuvables pour le swap.")