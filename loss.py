# loss.py
import torch
import torch.nn.functional as F

def loss_regression_paper(
    recon_x, x, 
    mu, logvar, 
    y_pred, y_true, 
    prior_mu, prior_logvar, 
    gamma=1.0,beta=0.005
):
    """
    Implémentation de l'Equation (4) du papier CCVAE adaptée à la régression.
    
    L_sup = E_q[ log p(x|z) ] - KL( q(z|x) || p(z|y) ) + alpha * log q(y|z_c)
    
    Args:
        gamma: Correspond au terme alpha dans le papier (poids de la supervision).
    """
    batch_size = x.size(0)

    # ----------------------------------------------------------
    # 1. log p(x|z) : Reconstruction
    # ----------------------------------------------------------
    # On utilise la BCE sommée sur tous les pixels (standard VAE)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # ----------------------------------------------------------
    # 2. Terme KL : KL( q(z|x) || p(z|y) )
    # ----------------------------------------------------------
    # C'est ici que ça diffère du VAE standard.
    # Le prior n'est pas N(0,1) partout.
    # - Pour z_c     : Prior = N(prior_mu, prior_logvar)  <-- Vient du Prior Network
    # - Pour z_not_c : Prior = N(0, 1)                    <-- Standard
    
    # On découpe les latents de l'encodeur
    z_c_dim = prior_mu.size(1)
    mu_c, mu_not = mu[:, :z_c_dim], mu[:, z_c_dim:]
    logvar_c, logvar_not = logvar[:, :z_c_dim], logvar[:, z_c_dim:]

    # A. KL pour z_not_c (Style) vs N(0,1)
    # Formule analytique : -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_not_c = -0.5 * torch.sum(1 + logvar_not - mu_not.pow(2) - logvar_not.exp())

    # B. KL pour z_c (Age) vs N(prior_mu, prior_logvar)
    # Formule analytique KL entre deux gaussiennes
    # KL( N0 || N1 ) = 0.5 * [ (var0 + (mu0-mu1)^2)/var1 - 1 + log(var1/var0) ]
    
    var_c = torch.exp(logvar_c)       # Variance encodeur
    p_var = torch.exp(prior_logvar)   # Variance prior
    
    # Terme par terme pour la stabilité
    # log(var1/var0) = log(var1) - log(var0) = p_logvar - logvar_c
    term_log = prior_logvar - logvar_c
    term_trace = (var_c + (mu_c - prior_mu).pow(2)) / (p_var + 1e-8)
    
    kld_c = 0.5 * torch.sum(term_log + term_trace - 1)

    # Total KL Divergence
    # Le signe est positif ici car on minimise la Loss (donc on minimise la KL)
    # Dans l'ELBO (à maximiser), ce serait un signe moins.
    kld_loss =beta*(kld_not_c + kld_c)

    # ----------------------------------------------------------
    # 3. log q(y|z_c) : Régression Latente
    # ----------------------------------------------------------
    # Maximiser la log-vraisemblance gaussienne revient à minimiser la MSE.
    # log q(y|z) ~ -||y - y_pred||^2
    # gamma correspond au alpha du papier.
    reg_loss = F.mse_loss(y_pred, y_true, reduction='sum')
    # reg_loss = F.smooth_l1_loss(y_pred, y_true, reduction='sum')
    # ----------------------------------------------------------
    # TOTAL LOSS (à minimiser)
    # ----------------------------------------------------------
    # Loss = -ELBO
    # Loss = Reconstruction_Error + KL_Divergence + Gamma * Regression_Error
    total_loss = recon_loss + kld_loss + (gamma * reg_loss)

    # Normalisation par batch_size pour l'affichage (optionnel mais recommandé)
    avg_loss = total_loss / batch_size

    stats = {
        "loss": avg_loss.item(),
        "recon": recon_loss.item() / batch_size,
        "kld": kld_loss.item() / batch_size,
        "reg": reg_loss.item() / batch_size
    }

    return total_loss, stats