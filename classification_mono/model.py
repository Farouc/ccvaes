import torch
import torch.nn as nn
import torch.nn.functional as F


class CCVAE(nn.Module):
    """
    Conditional Contrastive Variational Autoencoder (CCVAE)
    for single-attribute classification (e.g. hair color).
    """

    def __init__(
        self,
        img_channels=3,
        z_c_dim=16,
        z_not_c_dim=32,
        num_classes=10,
    ):
        """
        Args:
            img_channels (int): Number of image channels.
            z_c_dim (int): Dimension of the attribute-specific latent z_c.
            z_not_c_dim (int): Dimension of the nuisance / style latent z_not_c.
            num_classes (int): Number of classes for the attribute
                               (e.g. number of hair colors).
        """
        super().__init__()

        self.z_c_dim = z_c_dim
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = z_c_dim + z_not_c_dim
        self.num_classes = num_classes

        # --------------------------------------------------
        # 1. Encoder q(z | x)
        # --------------------------------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64 * 4 * 4, self.total_z_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, self.total_z_dim)

        # --------------------------------------------------
        # 2. Decoder p(x | z)
        # --------------------------------------------------
        self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

        # --------------------------------------------------
        # 3. Classifier q(y | z_c)
        # --------------------------------------------------
        self.classifier = nn.Linear(z_c_dim, num_classes)

        # --------------------------------------------------
        # 4. Conditional prior p(z_c | y)
        # --------------------------------------------------
        self.y_embedding = nn.Linear(num_classes, 32)
        self.cond_prior_mu = nn.Linear(32, z_c_dim)
        self.cond_prior_logvar = nn.Linear(32, z_c_dim)

    # --------------------------------------------------
    # Reparameterization trick
    # --------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x, y=None):
        """
        Args:
            x (Tensor): Input images, shape (B, C, H, W)
            y (Tensor, optional): Attribute labels, shape (B,)

        Returns:
            recon_x (Tensor): Reconstructed images
            mu (Tensor): Encoder mean
            logvar (Tensor): Encoder log-variance
            y_logits (Tensor): Classification logits
            prior_mu (Tensor or None): Conditional prior mean p(z_c | y)
            prior_logvar (Tensor or None): Conditional prior log-variance
        """

        # --------------------------------------------------
        # A. Encode
        # --------------------------------------------------
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # --------------------------------------------------
        # B. Sample z
        # --------------------------------------------------
        z = self.reparameterize(mu, logvar)

        # --------------------------------------------------
        # C. Split latent variables
        # --------------------------------------------------
        z_c = z[:, :self.z_c_dim]

        # --------------------------------------------------
        # D. Decode
        # --------------------------------------------------
        dec_in = self.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = self.decoder_conv(dec_in)

        # --------------------------------------------------
        # E. Classifier q(y | z_c)
        # (use encoder mean for stability)
        # --------------------------------------------------
        mu_c = mu[:, :self.z_c_dim]
        y_logits = self.classifier(mu_c)

        # --------------------------------------------------
        # F. Conditional prior p(z_c | y)
        # --------------------------------------------------
        prior_mu, prior_logvar = None, None
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_embed = self.y_embedding(y_onehot)
            prior_mu = self.cond_prior_mu(y_embed)
            prior_logvar = self.cond_prior_logvar(y_embed)

        return recon_x, mu, logvar, y_logits, prior_mu, prior_logvar
