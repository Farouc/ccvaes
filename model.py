import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE(nn.Module):
    def __init__(self, img_channels=3, z_c_dims=[16, 16], z_not_c_dim=32, num_classes_list=[10, 11]):
        """
        Args:
            z_c_dims (list): Liste des dimensions pour chaque sous-z_c. 
                             Ex: [16, 16] si on veut 16 dims pour hair et 16 pour face.
            num_classes_list (list): Liste du nombre de classes pour chaque attribut.
                                     Ex: [10, 11] (10 couleurs cheveux, 11 couleurs visage).
        """
        super(CCVAE, self).__init__()

        self.z_c_dims = z_c_dims
        self.num_classes_list = num_classes_list
        self.num_attributes = len(num_classes_list)
        
        # Dimensions totales
        self.total_z_c_dim = sum(z_c_dims)
        self.z_not_c_dim = z_not_c_dim
        self.total_z_dim = self.total_z_c_dim + z_not_c_dim

        # --------------------------
        # 1. ENCODER q(z|x) (Global)
        # --------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 4 * 4, self.total_z_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, self.total_z_dim)

        # --------------------------
        # 2. DECODER p(x|z) (Global)
        # --------------------------
        self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )

        # --------------------------
        # 3. CLASSIFIERS & PRIORS
        # --------------------------
        self.classifiers = nn.ModuleList()
        self.priors_embedding = nn.ModuleList()
        self.priors_mu = nn.ModuleList()
        self.priors_logvar = nn.ModuleList()

        for dim_z, num_cls in zip(z_c_dims, num_classes_list):
            self.classifiers.append(nn.Linear(dim_z, num_cls))
            
            self.priors_embedding.append(nn.Linear(num_cls, 32)) 
            self.priors_mu.append(nn.Linear(32, dim_z))
            self.priors_logvar.append(nn.Linear(32, dim_z))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        """
        y: Tensor de shape (Batch, NumAttributes)
        """
        # --- A. Encodeur ---
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # =========================================================
        # <--- AJOUT DROPOUT ICI (Le "Hard Mode" pour le modèle)
        # =========================================================
        if self.training:
            # On sépare la partie couleur (z_c) de la partie style (z_not_c)
            z_c = z[:, :self.total_z_c_dim]
            z_not_c = z[:, self.total_z_c_dim:]
            
            # On applique le Dropout sur z_c uniquement.
            # p=0.25 signifie qu'on met à zéro 25% des infos de couleur aléatoirement.
            # Cela force le décodeur à regarder z_not_c pour compenser.
            z_c = F.dropout(z_c, p=0.25)
            
            # On recolle les morceaux
            z = torch.cat([z_c, z_not_c], dim=1)
        # =========================================================

        # --- B. Slicing pour les classifieurs ---
        # Note : On ré-utilise le z potentiellement "dropouté" pour le décodeur, 
        # mais pour les classifieurs, on préfère souvent utiliser mu (propre) ou z avant dropout.
        # Ici, ton architecture originale utilisait 'mu' pour le classifieur (dans la loss), 
        # ce qui est très bien.
        # Attention : si tu utilises 'z' ici pour autre chose, sache qu'il est bruité.

        split_sizes = self.z_c_dims + [self.z_not_c_dim]
        # Pour les classifieurs et priors, on renvoie les mu originaux (non dropoutés)
        mu_parts = torch.split(mu, split_sizes, dim=1)
        mu_c_list = mu_parts[:-1]

        y_logits_list = []
        prior_mu_list = []
        prior_logvar_list = []

        # --- C. Classifieurs & Priors ---
        for i in range(self.num_attributes):
            logits = self.classifiers[i](mu_c_list[i])
            y_logits_list.append(logits)

            if y is not None:
                y_i = y[:, i]
                num_cls = self.num_classes_list[i]
                y_onehot = F.one_hot(y_i, num_classes=num_cls).float()
                
                embed = F.relu(self.priors_embedding[i](y_onehot))
                p_mu = self.priors_mu[i](embed)
                p_logvar = self.priors_logvar[i](embed)
                
                prior_mu_list.append(p_mu)
                prior_logvar_list.append(p_logvar)
            else:
                prior_mu_list.append(None)
                prior_logvar_list.append(None)

        # --- D. Reconstruction ---
        # Ici 'z' contient le dropout si on est en training
        dec_in = self.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = self.decoder_conv(dec_in)

        return recon_x, mu, logvar, y_logits_list, prior_mu_list, prior_logvar_list
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CCVAE(nn.Module):
#     def __init__(self, img_channels=3, z_c_dims=[16, 16], z_not_c_dim=32, num_classes_list=[10, 11]):
#         """
#         Args:
#             z_c_dims (list): Liste des dimensions pour chaque sous-z_c. 
#                              Ex: [16, 16] si on veut 16 dims pour hair et 16 pour face.
#             num_classes_list (list): Liste du nombre de classes pour chaque attribut.
#                                      Ex: [10, 11] (10 couleurs cheveux, 11 couleurs visage).
#         """
#         super(CCVAE, self).__init__()

#         self.z_c_dims = z_c_dims
#         self.num_classes_list = num_classes_list
#         self.num_attributes = len(num_classes_list)
        
#         # Dimensions totales
#         self.total_z_c_dim = sum(z_c_dims)
#         self.z_not_c_dim = z_not_c_dim
#         self.total_z_dim = self.total_z_c_dim + z_not_c_dim

#         # --------------------------
#         # 1. ENCODER q(z|x) (Global)
#         # --------------------------
#         # L'encodeur voit l'image entière et produit un gros vecteur latent
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
#             nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
#             nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
#             nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
#             nn.Flatten()
#         )

#         # Les têtes mu/logvar produisent TOUT (z_c1 + z_c2 + ... + z_not_c)
#         self.fc_mu = nn.Linear(64 * 4 * 4, self.total_z_dim)
#         self.fc_logvar = nn.Linear(64 * 4 * 4, self.total_z_dim)

#         # --------------------------
#         # 2. DECODER p(x|z) (Global)
#         # --------------------------
#         self.decoder_input = nn.Linear(self.total_z_dim, 64 * 4 * 4)
#         self.decoder_conv = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
#             nn.Sigmoid()
#         )

#         # --------------------------
#         # 3. CLASSIFIERS & PRIORS (Spécifiques par attribut)
#         # --------------------------
#         # On utilise ModuleList pour itérer dessus proprement
#         self.classifiers = nn.ModuleList()
#         self.priors_embedding = nn.ModuleList()
#         self.priors_mu = nn.ModuleList()
#         self.priors_logvar = nn.ModuleList()

#         for dim_z, num_cls in zip(z_c_dims, num_classes_list):
#             # Classifieur : z_c_i -> logits_i
#             self.classifiers.append(nn.Linear(dim_z, num_cls))
            
#             # Prior : y_i (onehot) -> embedding -> mu/logvar
#             # On projette le one-hot vers une couche cachée (ex: 32)
#             self.priors_embedding.append(nn.Linear(num_cls, 32)) 
#             self.priors_mu.append(nn.Linear(32, dim_z))
#             self.priors_logvar.append(nn.Linear(32, dim_z))

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x, y=None):
#         """
#         y: Tensor de shape (Batch, NumAttributes). Ex: [[4, 8], [2, 0], ...]
#         """
#         batch_size = x.size(0)

#         # --- A. Encodeur ---
#         h = self.encoder_conv(x)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         z = self.reparameterize(mu, logvar)

#         # --- B. Découpage du latent (Slicing) ---
#         # On doit séparer [z_c1, z_c2, ..., z_not_c]
#         # On prépare les tailles des sections pour torch.split
#         split_sizes = self.z_c_dims + [self.z_not_c_dim]
#         z_parts = torch.split(z, split_sizes, dim=1)
        
#         # Les parties z_c sont toutes sauf la dernière
#         z_c_list = z_parts[:-1] 
#         # z_not_c est la dernière partie
#         # z_not_c = z_parts[-1] 

#         # --- C. Classifieurs & Priors (Boucle sur les attributs) ---
#         y_logits_list = []
#         prior_mu_list = []
#         prior_logvar_list = []

#         # On a besoin des moyennes (mu) découpées aussi pour le classifieur (pas le sample z)
#         mu_parts = torch.split(mu, split_sizes, dim=1)
#         mu_c_list = mu_parts[:-1]

#         for i in range(self.num_attributes):
#             # 1. Classification : on prend le z_c_i correspondant (version moyenne mu)
#             logits = self.classifiers[i](mu_c_list[i])
#             y_logits_list.append(logits)

#             # 2. Prior p(z_c|y) si y est fourni
#             if y is not None:
#                 # On récupère le label i pour tout le batch
#                 y_i = y[:, i] # Shape (Batch,)
#                 num_cls = self.num_classes_list[i]
                
#                 # One-hot
#                 y_onehot = F.one_hot(y_i, num_classes=num_cls).float()
                
#                 # Reseau de Prior
#                 embed = F.relu(self.priors_embedding[i](y_onehot))
#                 p_mu = self.priors_mu[i](embed)
#                 p_logvar = self.priors_logvar[i](embed)
                
#                 prior_mu_list.append(p_mu)
#                 prior_logvar_list.append(p_logvar)
#             else:
#                 prior_mu_list.append(None)
#                 prior_logvar_list.append(None)

#         # --- D. Reconstruction ---
#         # Le décodeur prend tout le vecteur z concaténé (ce qui est déjà 'z' ici)
#         # Mais pour être sûr de la cohérence, on utilise z directement calculé plus haut
#         dec_in = self.decoder_input(z)
#         dec_in = dec_in.view(-1, 64, 4, 4)
#         recon_x = self.decoder_conv(dec_in)

#         # On retourne des LISTES pour les logits et les priors
#         return recon_x, mu, logvar, y_logits_list, prior_mu_list, prior_logvar_list