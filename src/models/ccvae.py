import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE(nn.Module):
    def __init__(self, 
                 img_channels=3, 
                 img_size=64,
                 z_c_dims=[16], 
                 z_not_c_dim=32, 
                 task_types=['classification'], # List: 'classification' or 'regression'
                 num_classes_or_dim=[10],       # List: num_classes (int) or output_dim (1 for regression)
                 dropout_p=0.0):
        """
        Unified CCVAE class handling Classification (Multi/Single) and Regression.
        
        Args:
            img_channels (int): 3 for RGB.
            img_size (int): 64 or 128. Determines the depth of the encoder.
            z_c_dims (list[int]): Dimensions of the latent sub-spaces for attributes.
            z_not_c_dim (int): Dimension of the style/identity latent space.
            task_types (list[str]): List of tasks per attribute, e.g., ['classification', 'regression'].
            num_classes_or_dim (list[int]): Number of classes (for classif) or 1 (for regression).
            dropout_p (float): Probability to zero out z_c during training (Disentanglement trick).
        """
        super(CCVAE, self).__init__()

        self.img_size = img_size
        self.z_c_dims = z_c_dims
        self.z_not_c_dim = z_not_c_dim
        self.total_z_c_dim = sum(z_c_dims)
        self.total_z_dim = self.total_z_c_dim + z_not_c_dim
        self.task_types = task_types
        self.num_classes_or_dim = num_classes_or_dim
        self.dropout_p = dropout_p
        self.num_attributes = len(z_c_dims)

        # --------------------------
        # 1. Dynamic Backbone Builder
        # --------------------------
        if img_size == 64:
            # Lighter architecture (Standard for CartoonSet)
            self.encoder_conv, flatten_dim = self._build_64_encoder(img_channels)
            self.decoder_conv = self._build_64_decoder(img_channels)
            self.decoder_input_dim = 64 * 4 * 4 # 1024
            self.decoder_reshape = (64, 4, 4)
        elif img_size == 128:
            # Heavier architecture (Standard for UTKFace/Regression)
            self.encoder_conv, flatten_dim = self._build_128_encoder(img_channels)
            self.decoder_conv = self._build_128_decoder(img_channels)
            self.decoder_input_dim = 512 * 4 * 4 # 8192
            self.decoder_reshape = (512, 4, 4)
        else:
            raise ValueError("img_size must be 64 or 128")

        # Variational Heads
        self.fc_mu = nn.Linear(flatten_dim, self.total_z_dim)
        self.fc_logvar = nn.Linear(flatten_dim, self.total_z_dim)
        self.decoder_input = nn.Linear(self.total_z_dim, self.decoder_input_dim)

        # --------------------------
        # 2. Attribute Heads (Classifiers / Regressors)
        # --------------------------
        self.heads = nn.ModuleList()
        self.priors_embedding = nn.ModuleList() # Only for classification
        self.priors_net = nn.ModuleList()       # Shared logic for prior mapping
        self.priors_mu = nn.ModuleList()
        self.priors_logvar = nn.ModuleList()

        for i, (dim_z, task, out_dim) in enumerate(zip(z_c_dims, task_types, num_classes_or_dim)):
            
            # --- Prediction Head (z_c -> y_pred) ---
            if task == 'classification':
                # Linear Classifier
                self.heads.append(nn.Linear(dim_z, out_dim))
            elif task == 'regression':
                # MLP Regressor (Stronger)
                self.heads.append(nn.Sequential(
                    nn.Linear(dim_z, 64), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Linear(64, 64),    nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Linear(64, out_dim)
                ))
            
            # --- Prior Network (y -> z_c_prior) ---
            # Step A: Embed y
            if task == 'classification':
                # Discrete Embedding: OneHot -> Dense
                self.priors_embedding.append(nn.Linear(out_dim, 32)) 
            elif task == 'regression':
                # Continuous Mapping: Scalar -> Dense
                self.priors_embedding.append(nn.Sequential(
                    nn.Linear(out_dim, 32), nn.ReLU(),
                    nn.Linear(32, 32) # Extra depth for regression prior
                ))

            # Step B: Map to Mu/Logvar (Shared structure)
            self.priors_mu.append(nn.Linear(32, dim_z))
            self.priors_logvar.append(nn.Linear(32, dim_z))

    # ==========================
    # Builders (Private methods)
    # ==========================
    def _build_64_encoder(self, c_in):
        net = nn.Sequential(
            nn.Conv2d(c_in, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),   nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),   nn.ReLU(),
            nn.Flatten()
        )
        return net, 64 * 4 * 4

    def _build_64_decoder(self, c_out):
        return nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, c_out, 4, 2, 1), nn.Sigmoid()
        )

    def _build_128_encoder(self, c_in):
        # Uses BatchNorm + LeakyReLU (Better for larger images/regression)
        layers = []
        channels = [c_in, 32, 64, 128, 256, 512]
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 4, 2, 1))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers), 512 * 4 * 4

    def _build_128_decoder(self, c_out):
        layers = []
        # Input is 512 x 4 x 4
        # Sequence: 512->256->128->64->32->c_out
        ch = [512, 256, 128, 64, 32]
        for i in range(len(ch)-1):
            layers.append(nn.ConvTranspose2d(ch[i], ch[i+1], 4, 2, 1))
            layers.append(nn.BatchNorm2d(ch[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer
        layers.append(nn.ConvTranspose2d(32, c_out, 4, 2, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        # 1. Encode
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # 2. Latent Splitting
        split_sizes = self.z_c_dims + [self.z_not_c_dim]
        
        # Use mu for classification/regression (cleaner signal), z for reconstruction
        mu_parts = torch.split(mu, split_sizes, dim=1)
        z_parts = torch.split(z, split_sizes, dim=1)
        
        mu_c_list = mu_parts[:-1]
        z_c_list = list(z_parts[:-1]) # Mutable list
        z_not_c = z_parts[-1]

        # 3. Dropout Strategy (Disentanglement "Hard Mode")
        if self.training and self.dropout_p > 0:
            for i in range(len(z_c_list)):
                z_c_list[i] = F.dropout(z_c_list[i], p=self.dropout_p)
        
        # Reassemble z for decoder
        z_dropped = torch.cat(z_c_list + [z_not_c], dim=1)

        # 4. Heads (Classification/Regression) & Priors
        outputs_pred = []
        outputs_prior_mu = []
        outputs_prior_logvar = []

        for i in range(self.num_attributes):
            # A. Predict y from z_c (using mu is standard)
            # If regression, output is value. If classif, output is logits.
            pred = self.heads[i](mu_c_list[i]) 
            outputs_pred.append(pred)

            # B. Conditional Prior p(z_c|y)
            if y is not None:
                # Get labels for this attribute
                y_i = y[:, i] 
                
                if self.task_types[i] == 'classification':
                    # One-hot encoding for classification
                    y_in = F.one_hot(y_i.long(), num_classes=self.num_classes_or_dim[i]).float()
                    embed = F.relu(self.priors_embedding[i](y_in))
                else:
                    # Direct value for regression (ensure it's float 2D tensor)
                    y_in = y_i.view(-1, 1).float() 
                    # Sequential already contains ReLUs for regression path
                    embed = self.priors_embedding[i](y_in)

                p_mu = self.priors_mu[i](embed)
                p_logvar = self.priors_logvar[i](embed)
                
                outputs_prior_mu.append(p_mu)
                outputs_prior_logvar.append(p_logvar)
            else:
                outputs_prior_mu.append(None)
                outputs_prior_logvar.append(None)

        # 5. Decode
        dec_in = self.decoder_input(z_dropped)
        dec_in = dec_in.view(-1, *self.decoder_reshape)
        recon_x = self.decoder_conv(dec_in)

        return recon_x, mu, logvar, outputs_pred, outputs_prior_mu, outputs_prior_logvar