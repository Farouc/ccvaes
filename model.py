import torch
import torch.nn as nn
import torch.nn.functional as F


class CCVAE(nn.Module):
    def __init__(self, img_channels=3, z_c_dim=16, z_not_c_dim=32, num_classes=10):
        """
        num_classes = nombre de couleurs de cheveux dans le dataset (ex: 10)
        """
        super(CCVAE, self).__init__()

        self.z_c_dim = z_c_dim
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = z_c_dim + z_not_c_dim
        self.num_classes = num_classes

        # --------------------------
        # 1. ENCODER q(z|x)
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        # self.fc_mu = nn.Linear(64 * 4 * 4, self.total_z_dim)
        # self.fc_logvar = nn.Linear(64 * 4 * 4, self.total_z_dim)

        self.fc_mu = nn.Linear(64 * 8 * 8, self.total_z_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, self.total_z_dim)


        # --------------------------
        # 2. DECODER p(x|z)
        # --------------------------
        # self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
        self.decoder_input = nn.Linear(self.total_z_dim, 64 * 8 * 8)

        # self.decoder_conv = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
        #     nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
        #     nn.Sigmoid()
        # )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),   # 8 → 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 16 → 32
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),   # 32 → 64
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),   # 64 → 128
            nn.ConvTranspose2d(16, img_channels, 3, 1, 1),
            nn.Sigmoid()
        )

        # --------------------------
        # 3. CLASSIFIER q(y|z_c)
        #    prédiction hair_color
        # --------------------------
        self.classifier = nn.Linear(z_c_dim, num_classes)

        # --------------------------
        # 4. CONDITIONAL PRIOR p(z_c|y)
        #    y_onehot → latent prior
        # --------------------------
        self.y_embedding = nn.Linear(num_classes, 32)  # projection du one-hot
        self.cond_prior_mu = nn.Linear(32, z_c_dim)
        self.cond_prior_logvar = nn.Linear(32, z_c_dim)

    # --------------------------
    # Reparameterization trick
    # --------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------
    # Forward pass complet
    # --------------------------
    def forward(self, x, y=None):
        # A. Encodeur
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # B. Sampling
        z = self.reparameterize(mu, logvar)

        # C. Split
        z_c = z[:, :self.z_c_dim]

        # D. Reconstruction
        dec_in = self.decoder_input(z)
        # dec_in = dec_in.view(-1, 64, 4, 4)
        dec_in = dec_in.view(-1, 64, 8, 8)

        recon_x = self.decoder_conv(dec_in)

        # E. Classifieur q(y|z_c)
        mu_c = mu[:, :self.z_c_dim]
        y_logits = self.classifier(mu_c)   # logits shape : (B, num_classes)

        # F. Conditional prior p(z_c|y)
        prior_mu, prior_logvar = None, None
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_embed = self.y_embedding(y_onehot)
            prior_mu = self.cond_prior_mu(y_embed)
            prior_logvar = self.cond_prior_logvar(y_embed)

        return recon_x, mu, logvar, y_logits, prior_mu, prior_logvar
