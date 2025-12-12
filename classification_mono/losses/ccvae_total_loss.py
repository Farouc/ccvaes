# losses/ccvae_total.py
import torch
from losses.elbo_ccvae import ccvae_elbo_supervised
from losses.contrastive_loss import supervised_contrastive_loss


def ccvae_supervised_loss(
    model,
    x,
    y,
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
    contrastive_weight=0.0,
    contrastive_on="mu",  # "mu" or "z"
):
    """
    Total supervised loss:
        L = -ELBO + λ * L_contrastive
    """

    elbo, elbo_stats = ccvae_elbo_supervised(
        model,
        x,
        y,
        K=K,
        recon_type=recon_type,
        uniform_class_prior=uniform_class_prior,
    )

    loss = -elbo
    loss_contrastive = torch.tensor(0.0, device=x.device)

    if contrastive_weight > 0:
        h = model.encoder_conv(x)
        mu = model.fc_mu(h)
        z_c = mu[:, :model.z_c_dim]
        loss_contrastive = supervised_contrastive_loss(z_c, y)
        loss = loss + contrastive_weight * loss_contrastive

    stats = {
        **elbo_stats,
        "loss": loss.item(),
        "loss_contrastive": loss_contrastive.item(),
    }

    return loss, stats
