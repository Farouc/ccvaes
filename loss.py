# loss.py
import torch
import torch.nn.functional as F


# ==============================================================
#  CCVAE SUPERVISED LOSS
#  (Utilisée quand x est accompagné de son label y)
# ==============================================================

def ccvae_loss_supervised(
    recon_x, x,
    mu, logvar,
    y_logits, y_true,
    prior_mu, prior_logvar,
    z_c_dim,
    recon_type="mse"
):
    """
    CCVAE supervised loss for a single categorical attribute.
    
    Components:
        1. Reconstruction loss
        2. KL(z_not_c || N(0,I))
        3. KL(z_c || p(z_c | y))
        4. Classification loss (CrossEntropy)
    """

    # ----------------------------------------------------------
    # 1. Reconstruction loss p(x|z)
    # ----------------------------------------------------------
    if recon_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # ----------------------------------------------------------
    # Split latent space
    # ----------------------------------------------------------
    mu_c      = mu[:, :z_c_dim]
    logvar_c  = logvar[:, :z_c_dim]
    mu_not_c  = mu[:, z_c_dim:]
    logvar_not_c = logvar[:, z_c_dim:]

    # ----------------------------------------------------------
    # 2. KL divergence for z_not_c against N(0, I)
    # ----------------------------------------------------------
    kl_not_c = -0.5 * torch.sum(
        1 + logvar_not_c - mu_not_c.pow(2) - logvar_not_c.exp()
    )

    # ----------------------------------------------------------
    # 3. KL divergence for z_c against conditional prior
    #    KL(q(z_c|x) || p(z_c | y))
    # ----------------------------------------------------------
    var_c     = logvar_c.exp()
    var_prior = prior_logvar.exp()

    kl_c = 0.5 * torch.sum(
        prior_logvar - logvar_c +
        (var_c + (mu_c - prior_mu).pow(2)) / var_prior
        - 1
    )

    # ----------------------------------------------------------
    # 4. Classification loss q(y|z_c)
    # ----------------------------------------------------------
    classif_loss = F.cross_entropy(y_logits, y_true, reduction="sum")

    # ----------------------------------------------------------
    # Total supervised loss
    # ----------------------------------------------------------
    total_loss = recon_loss + kl_not_c + kl_c + classif_loss

    return total_loss, recon_loss, kl_not_c, kl_c, classif_loss



# ==============================================================
#  CCVAE UNSUPERVISED LOSS
#  (Utilisée quand x est SANS label)
# ==============================================================

def ccvae_loss_unsupervised(
    recon_x, x,
    mu, logvar,
    z_c_dim,
    recon_type="mse"
):
    """
    CCVAE unsupervised loss.
    
    Components:
        1. Reconstruction loss
        2. KL(z || N(0,I))
    
    No classification, no conditional prior.
    """

    # ----------------------------------------------------------
    # 1. Reconstruction loss p(x|z)
    # ----------------------------------------------------------
    if recon_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # ----------------------------------------------------------
    # 2. KL divergence for FULL latent space against N(0,I)
    # ----------------------------------------------------------
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = recon_loss + kl

    return total_loss, recon_loss, kl



# ==============================================================
#  OPTIONAL: WRAPPER TO COMBINE SUPERVISED + UNSUPERVISED LOSS
# ==============================================================

def ccvae_loss_mixed(
    outputs_labeled,
    outputs_unlabeled,
    y_true,
    z_c_dim,
    lambda_unsup=1.0,
    recon_type="mse"
):
    """
    Utility for combining supervised and unsupervised batches.

    Parameters:
        outputs_labeled   = (recon_x_l, mu_l, logvar_l, y_logits_l, prior_mu_l, prior_logvar_l)
        outputs_unlabeled = (recon_x_u, mu_u, logvar_u)
        y_true            = labels for labeled batch
        lambda_unsup      = scaling factor for unsupervised loss
    """

    # Unpack labeled outputs
    recon_l, mu_l, logvar_l, y_logits_l, prior_mu_l, prior_logvar_l = outputs_labeled

    # Unpack unlabeled outputs
    recon_u, mu_u, logvar_u = outputs_unlabeled

    # Compute supervised loss
    loss_sup, rec_sup, klnot_sup, klc_sup, cls_sup = ccvae_loss_supervised(
        recon_l, recon_l,    # **FIXED BELOW**
        mu_l, logvar_l,
        y_logits_l, y_true,
        prior_mu_l, prior_logvar_l,
        z_c_dim=z_c_dim,
        recon_type=recon_type
    )

    # Compute unsupervised loss
    loss_unsup, rec_unsup, kl_unsup = ccvae_loss_unsupervised(
        recon_u, recon_u,    # **FIXED BELOW**
        mu_u, logvar_u,
        z_c_dim=z_c_dim,
        recon_type=recon_type
    )

    total = loss_sup + lambda_unsup * loss_unsup

    return total
