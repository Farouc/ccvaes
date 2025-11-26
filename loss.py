# loss.py
import math
import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)

def log_normal_diag(z, mu, logvar):
    """
    Log densité d'une gaussienne diagonale N(mu, diag(exp(logvar))).
    Retourne un tenseur (B,)
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1
    )

def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation de log p(x|z).
    Retourne un tenseur (B,)
    """
    if recon_type == "bce":
        # BCE pour images [0,1]
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        # MSE
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)


# ==============================================================
#  CCVAE SUPERVISED LOSS — Multi-Label
# ==============================================================

def ccvae_loss_supervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    y,          # (B, num_attributes) -> ex: [[4, 8], [2, 0]...]
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Version Multi-Label de l'équation (4).
    L = E[ log p(x|z) + sum_i(log p(z_ci|yi)) + log p(z_not) - log q(z|x) - sum_i(log q(yi|z_ci)) ]
    """

    # Récupération des dimensions pour le slicing
    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    # 1. Encodage q(z|x)
    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    # Accumulateurs pour l'Importance Sampling
    log_p_xz_list = []
    log_p_zy_list = []    # log p(z|y) = log p(zn) + sum log p(zci|yi)
    log_qz_x_list = []
    
    # Pour q(y|z), on va stocker la somme des logs (log joint prob)
    log_qy_zc_total_list = [] 
    
    # Pour calculer le q(y|x) agrégé plus tard, on a besoin des probs individuelles
    # Structure : liste de K éléments, chacun contenant une liste de N_attr tenseurs (B,)
    qy_zc_individual_list = [] 

    for k in range(K):
        # --- A. Sampling z ~ q(z|x) ---
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        
        # --- B. Slicing ---
        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        # --- C. log p(x|z) ---
        dec_in = model.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type) # (B,)

        # --- D. Traitement par Attribut ---
        # On somme les logs des différents attributs
        log_p_zc_given_y_sum = 0
        log_qy_given_zc_sum = 0
        
        # On garde les probs brutes pour le calcul de w (poids)
        current_k_probs = [] 

        for i in range(num_attributes):
            # -- Prior p(z_ci | yi) --
            # On récupère le label i pour le batch
            y_i = y[:, i] # (B,)
            y_onehot = F.one_hot(y_i, num_classes=model.num_classes_list[i]).float()
            
            # Passage dans le réseau Prior spécifique à l'attribut i
            embed = F.relu(model.priors_embedding[i](y_onehot))
            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)
            
            # Log densité
            log_p_zc_given_y_sum += log_normal_diag(z_c_list[i], p_mu, p_logvar)

            # -- Classifieur q(yi | z_ci) --
            logits = model.classifiers[i](z_c_list[i])
            log_probs = F.log_softmax(logits, dim=-1)
            
            # On prend la log-prob du vrai label y_i
            log_qy_i = log_probs.gather(1, y_i.view(-1, 1)).squeeze(1) # (B,)
            
            log_qy_given_zc_sum += log_qy_i
            current_k_probs.append(log_qy_i.exp()) # On stocke la proba (pas le log)

        # -- Prior p(z_not) --
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)

        # Total log p(z|y)
        log_p_zy = log_p_zc_given_y_sum + log_p_znot

        # Total log q(z|x)
        log_qz_x = log_normal_diag(z, mu, logvar)

        # Storage
        log_p_xz_list.append(log_p_xz)
        log_p_zy_list.append(log_p_zy)
        log_qz_x_list.append(log_qz_x)
        log_qy_zc_total_list.append(log_qy_given_zc_sum)
        qy_zc_individual_list.append(current_k_probs)

    # ----------------------------------------------------------
    # Agrégation et Poids d'importance (Weights)
    # ----------------------------------------------------------
    
    # 1. Calcul de q(y|x) pour chaque attribut séparément
    # q(yi|x) ≈ mean_over_K( q(yi|z_ci) )
    # On a besoin de log q(y|x) total = sum_i log q(yi|x)
    
    log_q_y_x_total = 0
    weights_total = 1.0 # Produit des poids w_i
    
    # Pour chaque attribut i
    for i in range(num_attributes):
        # Récupérer les probs de l'attribut i pour les K samples
        # shape (K, B)
        probs_i_k = torch.stack([k_list[i] for k_list in qy_zc_individual_list], dim=0)
        
        # Moyenne sur K => q(yi|x)
        q_yi_x = probs_i_k.mean(dim=0) # (B,)
        log_q_yi_x = torch.log(q_yi_x + 1e-8)
        
        # --- CORRECTION ICI ---
        log_q_y_x_total += log_q_yi_x # C'était log_yi_x avant (typo)
        
        # Poids w_i = q(yi|z_ci) / q(yi|x)
        # On calcule le poids total w = w1 * w2 * ...
        w_i = probs_i_k / (q_yi_x.unsqueeze(0) + 1e-8)
        weights_total = weights_total * w_i

    # 2. Terme "Inner" (partie dans l'espérance)
    # Convertir listes en tenseurs (K, B)
    log_p_xz = torch.stack(log_p_xz_list, dim=0)
    log_p_zy = torch.stack(log_p_zy_list, dim=0)
    log_qz_x = torch.stack(log_qz_x_list, dim=0)
    log_qy_zc = torch.stack(log_qy_zc_total_list, dim=0)

    # log [ p(x|z)p(z|y) / (q(y|z_c) q(z|x)) ]
    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc

    # 3. Moyenne pondérée
    # E_{q(z|x)} [ w * inner ]
    weighted_inner = (weights_total * inner).mean(dim=0) # (B,)

    # 4. Final Loss
    # L(x,y) = Weighted_Inner + log q(y|x) + log p(y)
    # On néglige log p(y) si uniforme
    L_xy = weighted_inner + log_q_y_x_total

    elbo = L_xy.mean()
    loss = -elbo

    with torch.no_grad():
        stats = dict(
            loss=loss.item(),
            log_p_xz=log_p_xz.mean().item(),
            log_qy_zc=log_qy_zc.mean().item(),
        )

    return loss, stats


# ==============================================================
#  CCVAE UNSUPERVISED LOSS — Multi-Label
# ==============================================================

def ccvae_loss_unsupervised_paper(
    model,
    x,
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Version Multi-Label de l'équation (5).
    On sample y_i ~ q(y_i | z_c_i) pour chaque attribut.
    """
    # split_sizes = model.z_c_dims + [model.z_not_c_dim]
    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    log_terms_list = []

    for k in range(K):
        # --- A. Sample z ---
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        # --- B. Boucle Attributs (Sample y + Calc Prior/Classif) ---
        log_p_zc_given_y_sum = 0
        log_qy_given_zc_sum = 0
        
        for i in range(num_attributes):
            # 1. Prediction q(yi | zci)
            logits = model.classifiers[i](z_c_list[i])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            
            # 2. Sampling y_sample ~ Categorical(probs)
            y_sample = torch.multinomial(probs, num_samples=1).squeeze(1) # (B,)
            
            # Log q(y_sample | zci)
            log_qy_i = log_probs.gather(1, y_sample.view(-1, 1)).squeeze(1)
            log_qy_given_zc_sum += log_qy_i

            # 3. Prior p(zci | y_sample)
            y_onehot = F.one_hot(y_sample, num_classes=model.num_classes_list[i]).float()
            
            embed = F.relu(model.priors_embedding[i](y_onehot))
            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)
            
            log_p_zc_given_y_sum += log_normal_diag(z_c_list[i], p_mu, p_logvar)

        # --- C. Reste (Recon + z_not + qz) ---
        dec_in = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)

        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)

        log_qz_x = log_normal_diag(z, mu, logvar)

        # Somme finale
        log_p_zy = log_p_zc_given_y_sum + log_p_znot
        
        # Terme interne
        # log [ p(x|z)p(z|y)p(y) / (q(y|z_c) q(z|x)) ]
        # on ignore log p(y)
        inner = log_p_xz + log_p_zy - log_qy_given_zc_sum - log_qz_x
        log_terms_list.append(inner)

    # Moyenne sur K
    log_terms = torch.stack(log_terms_list, dim=0) # (K, B)
    L_x = log_terms.mean(dim=0) # (B,)

    elbo = L_x.mean()
    loss = -elbo

    stats = dict(
        loss=loss.item(),
        elbo=elbo.item()
    )

    return loss, stats