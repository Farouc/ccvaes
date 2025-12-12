import torch
import torch.nn as nn
import torch.nn.functional as F


class CCVAE(nn.Module):
    """
    Conditional Contrastive Variational Autoencoder (CCVAE)
    for multi-attribute classification on images.

    The latent space is decomposed as:
        z = [z_c_1, z_c_2, ..., z_c_K, z_not_c]

    where:
        - z_c_i encodes attribute i (e.g. hair color, face color)
        - z_not_c encodes shared nuisance / style information
    """

    def __init__(
        self,
        img_channels=3,
        z_c_dims=[16, 16],
        z_not_c_dim=32,
        num_classes_list=[10, 11],
    ):
        """
        Args:
            img_channels (int): Number of image channels.
            z_c_dims (list[int]): Latent dimensions for each attribute-specific z_c.
                                 Example: [16, 16]
            z_not_c_dim (int): Dimension of the nuisance latent z_not_c.
            num_classes_list (list[int]): Number of classes per attribute.
                                          Example: [10, 11]
        """
        super().__init__()

        self.z_c_dims = z_c_dims
        self.num_classes_list = num_classes_list
        self.num_attributes = len(num_classes_list)

        # Total latent dimensions
        self.total_z_c_dim = sum(z_c_dims)
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = self.total_z_c_dim + z_not_c_dim

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
        # 3. Attribute classifiers and conditional priors
        # --------------------------------------------------
        self.classifiers = nn.ModuleList()
        self.prior_embeddings = nn.ModuleList()
        self.prior_mu = nn.ModuleList()
        self.prior_logvar = nn.ModuleList()

        for dim_z, num_cls in zip(z_c_dims, num_classes_list):
            # q(y_i | z_c_i)
            self.classifiers.append(nn.Linear(dim_z, num_cls))

            # p(z_c_i | y_i)
            self.prior_embeddings.append(nn.Linear(num_cls, 32))
            self.prior_mu.append(nn.Linear(32, dim_z))
            self.prior_logvar.append(nn.Linear(32, dim_z))

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
            y (Tensor, optional): Attribute labels, shape (B, num_attributes)

        Returns:
            recon_x (Tensor): Reconstructed images
            mu (Tensor): Encoder mean
            logvar (Tensor): Encoder log-variance
            y_logits_list (list[Tensor]): Classifier logits per attribute
            prior_mu_list (list[Tensor]): Prior means p(z_c | y)
            prior_logvar_list (list[Tensor]): Prior log-variances p(z_c | y)
        """

        # --------------------------------------------------
        # A. Encode
        # --------------------------------------------------
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # --------------------------------------------------
        # B. Split latent variables
        # --------------------------------------------------
        split_sizes = self.z_c_dims + [self.z_not_c_dim]

        # Use encoder means (mu) for classification and priors
        mu_parts = torch.split(mu, split_sizes, dim=1)
        mu_c_list = mu_parts[:-1]

        # --------------------------------------------------
        # C. Attribute classifiers and conditional priors
        # --------------------------------------------------
        y_logits_list = []
        prior_mu_list = []
        prior_logvar_list = []

        for i in range(self.num_attributes):
            # Classifier q(y_i | z_c_i)
            logits = self.classifiers[i](mu_c_list[i])
            y_logits_list.append(logits)

            if y is not None:
                y_i = y[:, i]
                num_cls = self.num_classes_list[i]

                y_onehot = F.one_hot(y_i, num_classes=num_cls).float()
                embed = F.relu(self.prior_embeddings[i](y_onehot))

                p_mu = self.prior_mu[i](embed)
                p_logvar = self.prior_logvar[i](embed)

                prior_mu_list.append(p_mu)
                prior_logvar_list.append(p_logvar)
            else:
                prior_mu_list.append(None)
                prior_logvar_list.append(None)

        # --------------------------------------------------
        # D. Decode
        # --------------------------------------------------
        dec_in = self.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = self.decoder_conv(dec_in)

        return (
            recon_x,
            mu,
            logvar,
            y_logits_list,
            prior_mu_list,
            prior_logvar_list,
        )
