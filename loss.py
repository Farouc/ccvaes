import torch
import torch.nn.functional as F

def loss_function_ccvae(recon_x, x, mu, logvar, y_pred_logits, y_true, 
                        prior_mu, prior_logvar, z_c_dim):
    """
    Implémentation de l'équation (4) du papier pour le cas supervisé.
    """
    
    # 1. Reconstruction Loss (p(x|z))
    # Binary Cross Entropy si images [0,1], sinon MSE
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # Séparation des paramètres encoder pour z_c et z_not_c
    mu_c = mu[:, :z_c_dim]
    logvar_c = logvar[:, :z_c_dim]
    
    mu_not_c = mu[:, z_c_dim:]
    logvar_not_c = logvar[:, z_c_dim:]

    # 2. KL Divergence pour z_not_c (Latents de style)
    # On veut que z_not_c ressemble à N(0, I) -> KL Standard
    # Formule : -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_not_c = -0.5 * torch.sum(1 + logvar_not_c - mu_not_c.pow(2) - logvar_not_c.exp())

    # 3. KL Divergence pour z_c (Latents caractéristiques)
    # On veut que q(z_c|x) ressemble à p(z_c|y) (le conditional prior) [cite: 496]
    # Formule KL entre deux Gaussiennes générales :
    # log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2*sigma2^2) - 0.5
    
    # Var_enc = exp(logvar_c), Var_prior = exp(prior_logvar)
    var_c = logvar_c.exp()
    var_prior = prior_logvar.exp()
    
    kl_c = 0.5 * torch.sum(
        prior_logvar - logvar_c + 
        (var_c + (mu_c - prior_mu).pow(2)) / var_prior - 1
    )

    # 4. Classification Loss (log q(y|z_c)) 
    # Le papier mentionne un terme de classifieur explicite
    classif_loss = F.cross_entropy(y_pred_logits, y_true, reduction='sum')

    # Total Loss
    # On pondère souvent le KL par beta (ici 1.0 pour simplifier)
    total_loss = recon_loss + kl_not_c + kl_c + classif_loss
    
    return total_loss, recon_loss, kl_c, classif_loss