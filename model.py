import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE(nn.Module):
    def __init__(self, img_channels=3, z_c_dim=18, z_not_c_dim=27, num_labels=18):
        super(CCVAE, self).__init__()
        
        self.z_c_dim = z_c_dim         # Latents caractéristiques (liés aux labels)
        self.z_not_c_dim = z_not_c_dim # Latents contextuels (style, fond...)
        self.total_z_dim = z_c_dim + z_not_c_dim
        self.num_labels = num_labels

        # --------------------------
        # 1. ENCODER q(z|x)
        # Architecture simplifiée inspirée de Table 3 [cite: 446]
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        # Sortie conv pour 64x64 input -> 64 * 4 * 4 = 1024
        self.fc_mu = nn.Linear(1024, self.total_z_dim)
        self.fc_logvar = nn.Linear(1024, self.total_z_dim)

        # --------------------------
        # 2. DECODER p(x|z)
        # --------------------------
        self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid() # Pour des images entre [0, 1]
        )

        # --------------------------
        # 3. CLASSIFIER q(y|z_c)
        # Prend z_c et prédit y [cite: 441]
        # --------------------------
        self.classifier = nn.Linear(self.z_c_dim, num_labels)

        # --------------------------
        # 4. CONDITIONAL PRIOR p(z_c|y)
        # Prend y et prédit la distribution "idéale" de z_c (mu et sigma) [cite: 442]
        # --------------------------
        self.cond_prior_mu = nn.Linear(num_labels, self.z_c_dim)
        self.cond_prior_logvar = nn.Linear(num_labels, self.z_c_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        # A. Encodage
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # B. Sampling du latent complet z
        z = self.reparameterize(mu, logvar)
        
        # C. Séparation (Slicing) des latents
        # z_c = partie gauche, z_not_c = partie droite
        z_c = z[:, :self.z_c_dim]
        # Note: z_not_c n'est pas utilisé explicitement sauf pour reconstruction
        
        # D. Reconstruction
        dec_in = self.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = self.decoder_conv(dec_in)
        
        # E. Outputs auxiliaires (Classifier & Prior)
        # Pour le classifieur, le papier suggère d'utiliser mu ou z sans reparam
        # pour stabiliser les gradients (Eq 8 / Appendix C.3.1) [cite: 456]
        mu_c = mu[:, :self.z_c_dim]
        y_pred_logits = self.classifier(mu_c) 
        
        prior_mu, prior_logvar = None, None
        if y is not None:
            # Si on a les labels, on calcule le Conditional Prior
            prior_mu = self.cond_prior_mu(y.float())
            prior_logvar = self.cond_prior_logvar(y.float())

        return recon_x, mu, logvar, y_pred_logits, prior_mu, prior_logvar