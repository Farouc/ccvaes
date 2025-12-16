import torch
import torch.nn.functional as F

def ccvae_regression_loss(
    recon_x, x,            # Reconstruction and original input
    mu, logvar,            # Encoder latents q(z|x)
    y_pred, y_true,        # Regression output q(y|z_c) and true label
    prior_mu, prior_logvar,# Prior latents p(z|y) from Prior Network
    gamma=1.0, 
    beta=0.005
):
    """
    Implementation of the CCVAE (Conditional CVAE) Loss adapted for Regression (Age).
    
    The objective is to minimize the negative ELBO (Evidence Lower Bound), which is:
    Loss = -E_q[ log p(x|z) ] + KL( q(z|x) || p(z|y) ) + gamma * Regression_Error
    
    Args:
        recon_x (Tensor): Reconstructed image $\hat{x}$.
        x (Tensor): Original input image $x$.
        mu (Tensor): Mean of the variational posterior $q(z|x)$.
        logvar (Tensor): Log variance of the variational posterior $q(z|x)$.
        y_pred (Tensor): Predicted label $y$ from $q(y|z_c)$.
        y_true (Tensor): True label $y$.
        prior_mu (Tensor): Mean of the conditional prior $p(z_c|y)$.
        prior_logvar (Tensor): Log variance of the conditional prior $p(z_c|y)$.
        gamma (float): Weight for the supervised regression term (corresponds to $\alpha$ in the paper).
        beta (float): Weight for the KL Divergence term ($\beta$-VAE factor).
        
    Returns:
        total_loss (Tensor): The total scalar loss.
        stats (dict): Dictionary of un-normalized loss components.
    """
    batch_size = x.size(0)

    # ----------------------------------------------------------
    # 1. Negative Log-Likelihood (Reconstruction Loss)
    # ----------------------------------------------------------
    # Using Binary Cross-Entropy summed over all pixels (Standard VAE for image data)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # ----------------------------------------------------------
    # 2. KL Divergence: KL( q(z|x) || p(z|y) )
    # ----------------------------------------------------------
    # This is the conditional KL term. The latent space z is split into:
    # - z_c (Content/Age): Prior is conditional: N(prior_mu, exp(prior_logvar))
    # - z_not_c (Style): Prior is standard Gaussian: N(0, 1)
    
    # Split the encoder latents
    z_c_dim = prior_mu.size(1)
    mu_c, mu_not = mu[:, :z_c_dim], mu[:, z_c_dim:]
    logvar_c, logvar_not = logvar[:, :z_c_dim], logvar[:, z_c_dim:]

    # A. KL for z_not_c (Style) vs N(0,1)
    # KL( N(mu, logvar) || N(0, 1) ) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_not_c = -0.5 * torch.sum(1 + logvar_not - mu_not.pow(2) - logvar_not.exp())

    # B. KL for z_c (Content) vs N(prior_mu, prior_logvar)
    # KL( N0 || N1 ) = 0.5 * [ (var0 + (mu0-mu1)^2)/var1 - 1 + log(var1/var0) ]
    var_c = torch.exp(logvar_c)        # q(z_c|x) variance
    p_var = torch.exp(prior_logvar)    # p(z_c|y) variance
    
    # Term-by-term calculation for stability:
    term_log = prior_logvar - logvar_c
    term_trace = (var_c + (mu_c - prior_mu).pow(2)) / (p_var + 1e-8) # Add epsilon for stability
    
    kld_c = 0.5 * torch.sum(term_log + term_trace - 1)

    # Total KL Divergence (Scaled by beta)
    kld_loss = beta * (kld_not_c + kld_c)

    # ----------------------------------------------------------
    # 3. Supervised Regression Loss: gamma * log q(y|z_c)
    # ----------------------------------------------------------
    # Maximizing the Gaussian log-likelihood log q(y|z_c) is equivalent to 
    # minimizing the Mean Squared Error (MSE).
    reg_loss = F.mse_loss(y_pred, y_true, reduction='sum')
    
    # ----------------------------------------------------------
    # TOTAL LOSS (to minimize)
    # ----------------------------------------------------------
    # Loss = Recon_Error + KL_Divergence + Gamma * Regression_Error
    total_loss = recon_loss + kld_loss + (gamma * reg_loss)

    # Compute statistics, normalized by batch_size for logging
    stats = {
        "loss": total_loss.item() / batch_size,
        "recon": recon_loss.item() / batch_size,
        "kld": kld_loss.item() / batch_size,
        "reg": reg_loss.item() / batch_size
    }

    return total_loss, stats