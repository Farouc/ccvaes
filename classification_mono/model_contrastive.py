import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE(nn.Module):
    """
    Conditional Constrained Variational Autoencoder (CCVAE) 
    Architecture for 64x64 input images.
    """
    def __init__(self, img_channels=3, z_c_dim=16, z_not_c_dim=64, num_classes=10):
        """
        Args:
            img_channels (int): Number of image channels (3 for RGB).
            z_c_dim (int): Dimension of the supervised/conditional latent space (z_c).
            z_not_c_dim (int): Dimension of the unsupervised latent space (z_not_c).
            num_classes (int): Number of hair color classes.
        """
        super(CCVAE, self).__init__()

        self.z_c_dim = z_c_dim
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = z_c_dim + z_not_c_dim # 16 + 64 = 80
        self.num_classes = num_classes

        # The feature map size after the encoder for 64x64 input is 64x4x4 (1024 flat)
        FEATURE_MAP_SIZE = 64 * 4 * 4 

        # --------------------------
        # 1. ENCODER q(z|x)
        # Input: 64x64x3. Output: 4x4x64 (Flattened to 1024)
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(), # 64 -> 32
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),           # 32 -> 16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),           # 16 -> 8
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),           # 8 -> 4
            nn.Flatten()
        )

        # FC layers mapping 1024 features to latent means/logvars (80 dimensions total)
        self.fc_mu = nn.Linear(FEATURE_MAP_SIZE, self.total_z_dim)
        self.fc_logvar = nn.Linear(FEATURE_MAP_SIZE, self.total_z_dim)


        # --------------------------
        # 2. DECODER p(x|z)
        # Input: z (80 dim). Output: 64x64x3
        # --------------------------
        # Initial layer maps latent vector (80 dim) back to the spatial feature map size (1024)
        self.decoder_input = nn.Linear(self.total_z_dim, FEATURE_MAP_SIZE)

        # Deconvolutional layers (4x4x64 -> 64x64x3)
        # This is the 4-layer structure that matches the checkpoint
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),        # 4 -> 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),        # 8 -> 16
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),        # 16 -> 32
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),         # 32 -> 64 (Final layer to 3 channels)
            nn.Sigmoid()
        )

        # --------------------------
        # 3. CLASSIFIER q(y|z_c)
        # Prediction of the supervised attribute (hair_color)
        # --------------------------
        self.classifier = nn.Linear(z_c_dim, num_classes)

        # --------------------------
        # 4. CONDITIONAL PRIOR p(z_c|y)
        # --------------------------
        self.y_embedding = nn.Linear(num_classes, 32)
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
    # Forward pass
    # --------------------------
    def forward(self, x, y=None):
        # A. Encoder
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # B. Sampling
        z = self.reparameterize(mu, logvar)

        # C. Split (z_c: conditional, z_not_c: disentangled)
        z_c = z[:, :self.z_c_dim]

        # D. Reconstruction
        dec_in = self.decoder_input(z)
        # Reshape to 4x4 spatial size (matching the encoder output)
        dec_in = dec_in.view(-1, 64, 4, 4) 

        recon_x = self.decoder_conv(dec_in)

        # E. Classifier q(y|z_c)
        mu_c = mu[:, :self.z_c_dim]
        y_logits = self.classifier(mu_c)

        # F. Conditional prior p(z_c|y)
        prior_mu, prior_logvar = None, None
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_embed = self.y_embedding(y_onehot)
            prior_mu = self.cond_prior_mu(y_embed)
            prior_logvar = self.cond_prior_logvar(y_embed)

        return recon_x, mu, logvar, y_logits, prior_mu, prior_logvar