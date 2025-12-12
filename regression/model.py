import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE_Age(nn.Module):
    def __init__(self, img_channels=3, z_c_dim=16, z_not_c_dim=64):
        super(CCVAE_Age, self).__init__()
        
        self.z_c_dim = z_c_dim
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = z_c_dim + z_not_c_dim

        # --------------------------
        # 1. ENCODER (5 layers 128x128)
        # --------------------------
        self.encoder_conv = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(img_channels, 32, 4, 2, 1), # -> 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),           # -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),          # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),         # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),         # -> 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )

        # Bottleneck : 512 channels * 4 * 4 spatial
        hidden_dim = 512 * 4 * 4 
        self.fc_mu = nn.Linear(hidden_dim, self.total_z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.total_z_dim)

        # --------------------------
        # 2. DECODER (5 mirror layers)
        # --------------------------
        self.decoder_input = nn.Linear(self.total_z_dim, hidden_dim)
        
        self.decoder_conv = nn.Sequential(
            # Input reshaped: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Sortie
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1), # -> 3 x 128 x 128
            nn.Sigmoid()
        )

        # --------------------------
        # 3. REGRESSOR
        # --------------------------
        self.regressor = nn.Sequential(
            nn.Linear(z_c_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

        # --------------------------
        # 4. PRIOR NET
        # --------------------------
        self.prior_net = nn.Sequential(
            nn.Linear(1, 32),
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
        # A. Encodeur
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # B. Split
        z_c = z[:, :self.z_c_dim]
        
        # C. Regressor
        y_pred = self.regressor(z_c)
        
        # D. Prior
        p_mu, p_logvar = None, None
        if y is not None:
            h_prior = self.prior_net(y)
            p_mu = self.prior_mu(h_prior)
            p_logvar = self.prior_logvar(h_prior)
            
        # E. Decoder
        dec_in = self.decoder_input(z).view(-1, 512, 4, 4)
        recon_x = self.decoder_conv(dec_in)
        
        return recon_x, mu, logvar, y_pred, p_mu, p_logvar
