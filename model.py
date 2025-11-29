import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE_Age(nn.Module):
    def __init__(self, img_channels=3, z_c_dim=16, z_not_c_dim=64):
        """
        Modèle CCVAE adapté pour la régression (ex: Age).
        Args:
            img_channels (int): 3 pour RGB.
            z_c_dim (int): Dimension pour encoder l'âge (le concept).
            z_not_c_dim (int): Dimension pour encoder l'identité (le style).
        """
        super(CCVAE_Age, self).__init__()

        self.z_c_dim = z_c_dim
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = z_c_dim + z_not_c_dim

        # --------------------------
        # 1. ENCODER q(z|x)
        # --------------------------
        # Architecture standard pour 64x64
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        # Bottleneck size = 64 channels * 4 * 4 spatial size = 1024
        self.fc_mu = nn.Linear(64 * 4 * 4, self.total_z_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, self.total_z_dim)

        # --------------------------
        # 2. DECODER p(x|z)
        # --------------------------
        self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid() # Sortie entre 0 et 1
        )

        # --------------------------
        # 3. REGRESSOR q(y|z_c)
        # --------------------------
        # Remplace le Classifier.
        # Entrée : z_c (vecteur latent) -> Sortie : y_pred (Age estimé, scalaire)
        self.regressor = nn.Sequential(
            nn.Linear(z_c_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Une seule sortie continue
        )

        # --------------------------
        # 4. PRIOR NETWORK p(z_c|y)
        # --------------------------
        # Remplace l'Embedding.
        # Entrée : y (Age réel) -> Sortie : mu et logvar du prior
        self.prior_net = nn.Sequential(
            nn.Linear(1, 32),    # Prend une valeur scalaire (ex: 0.26)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(32, z_c_dim)
        self.prior_logvar = nn.Linear(32, z_c_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        """
        x: Images (Batch, 3, 64, 64)
        y: Ages normalisés (Batch, 1) optionnel
        """
        # --- A. Encodeur ---
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # --- B. Split (z_c vs z_not_c) ---
        z_c = z[:, :self.z_c_dim]
        z_not_c = z[:, self.z_c_dim:]

        # --- C. Regression (Auxiliary Task) ---
        # On prédit l'âge à partir du latent z_c
        y_pred = self.regressor(z_c)

        # --- D. Conditional Prior ---
        # Si on a l'âge réel (y), on calcule où z_c DEVRAIT être
        prior_mu, prior_logvar = None, None
        if y is not None:
            # y est de shape (Batch, 1)
            h_prior = self.prior_net(y)
            prior_mu = self.prior_mu(h_prior)
            prior_logvar = self.prior_logvar(h_prior)

        # --- E. Décodeur ---
        # Pour le training, on reconstruit l'image
        dec_in = self.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = self.decoder_conv(dec_in)

        return recon_x, mu, logvar, y_pred, prior_mu, prior_logvar