import torch
import torch.nn.functional as F


def loss_regression_paper(
    recon_x, x,
    mu, logvar,
    y_pred, y_true,
    prior_mu, prior_logvar,
    gamma=1.0,
    beta=0.005,
):
    """
    CCVAE regression loss corresponding to Eq. (4) of the CCVAE paper,
    adapted to a continuous target (regression).

    The objective corresponds to the negative ELBO:

        L = Reconstruction
            + KL( q(z|x) || p(z|y) )
            + gamma * Regression

    where:
        - z = (z_c, z_not_c)
        - p(z_c | y) is a learned conditional Gaussian prior
        - p(z_not_c) = N(0, I)

    Args:
        recon_x (Tensor): reconstructed images (B, C, H, W)
        x (Tensor): input images (B, C, H, W)
        mu (Tensor): encoder mean (B, D)
        logvar (Tensor): encoder log-variance (B, D)
        y_pred (Tensor): predicted target (B, 1)
        y_true (Tensor): ground-truth target (B, 1)
        prior_mu (Tensor): conditional prior mean for z_c (B, D_c)
        prior_logvar (Tensor): conditional prior log-variance for z_c (B, D_c)
        gamma (float): weight of the regression term (α in the paper)
        beta (float): weight of the KL divergence

    Returns:
        total_loss (Tensor): scalar loss to minimize
        stats (dict): dictionary of normalized loss components
    """

    batch_size = x.size(0)

    # ----------------------------------------------------------
    # 1. Reconstruction term: log p(x | z)
    # ----------------------------------------------------------
    # Binary Cross-Entropy summed over all pixels
    recon_loss = F.binary_cross_entropy(
        recon_x, x, reduction="sum"
    )

    # ----------------------------------------------------------
    # 2. KL divergence: KL( q(z|x) || p(z|y) )
    # ----------------------------------------------------------
    # The prior differs from a standard VAE:
    #   - z_c     ~ N(prior_mu, prior_logvar)
    #   - z_not_c ~ N(0, I)

    z_c_dim = prior_mu.size(1)

    # Split
