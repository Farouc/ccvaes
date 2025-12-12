# losses/elbo_ccvae.py
import math
import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def log_normal_diag(z, mu, logvar):
    """
    Log density of a diagonal Gaussian N(mu, diag(exp(logvar))).
    Returns (B,)
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1
    )


def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation of log p(x|z) (up to constant).
    Returns (B,)
    """
    if recon_type == "bce":
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)


# ==========================================================
#  CCVAE SUPERVISED ELBO — Eq. (4)
# ==========================================================

def ccvae_elbo_supervised(
    model,
    x,
    y,
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Pure CCVAE supervised ELBO (Eq. 4).
    Returns:
        elbo (scalar),
        stats (dict)
    """

    device = x.device
    B = x.size(0)
    z_c_dim = model.z_c_dim

    # q(z|x)
    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    # p(z_c|y)
    y_onehot = F.one_hot(y, num_classes=model.num_classes).float()
    y_embed = model.y_embedding(y_onehot)
    prior_mu = model.cond_prior_mu(y_embed)
    prior_logvar = model.cond_prior_logvar(y_embed)

    log_p_xz, log_p_zy, log_qz_x, log_qy_zc, qy_zc = [], [], [], [], []

    for _ in range(K):
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        # log p(x|z)
        dec = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec)
        log_p_xz_k = log_px_given_z(x, recon_x, recon_type)

        # log p(z|y)
        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)
        log_p_znot = log_normal_diag(
            z_not_c,
            torch.zeros_like(z_not_c),
            torch.zeros_like(z_not_c),
        )
        log_p_zy_k = log_p_zc_y + log_p_znot

        # log q(z|x)
        log_qz_x_k = log_normal_diag(z, mu, logvar)

        # q(y|z_c)
        logits = model.classifier(z_c)
        log_probs = F.log_softmax(logits, dim=-1)
        log_qy_zc_k = log_probs.gather(1, y.view(-1, 1)).squeeze(1)
        qy_zc_k = log_qy_zc_k.exp()

        log_p_xz.append(log_p_xz_k)
        log_p_zy.append(log_p_zy_k)
        log_qz_x.append(log_qz_x_k)
        log_qy_zc.append(log_qy_zc_k)
        qy_zc.append(qy_zc_k)

    log_p_xz = torch.stack(log_p_xz)
    log_p_zy = torch.stack(log_p_zy)
    log_qz_x = torch.stack(log_qz_x)
    log_qy_zc = torch.stack(log_qy_zc)
    qy_zc = torch.stack(qy_zc)

    q_y_x = qy_zc.mean(dim=0)
    log_q_y_x = torch.log(q_y_x + 1e-8)

    w = qy_zc / (q_y_x.unsqueeze(0) + 1e-8)
    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc
    weighted_inner = (w * inner).mean(dim=0)

    log_p_y = 0.0 if uniform_class_prior else 0.0
    L_xy = weighted_inner + log_q_y_x + log_p_y

    elbo = L_xy.mean()

    stats = {
        "elbo": elbo.item(),
        "log_p_xz": log_p_xz.mean().item(),
        "log_p_zy": log_p_zy.mean().item(),
        "log_qz_x": log_qz_x.mean().item(),
        "log_qy_zc": log_qy_zc.mean().item(),
        "log_q_y_x": log_q_y_x.mean().item(),
    }

    return elbo, stats


# ==========================================================
#  CCVAE UNSUPERVISED ELBO — Eq. (5)
# ==========================================================

def ccvae_elbo_unsupervised(
    model,
    x,
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Pure CCVAE unsupervised ELBO (Eq. 5).
    """

    z_c_dim = model.z_c_dim

    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    terms = []

    for _ in range(K):
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        logits = model.classifier(z_c)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        y = torch.multinomial(probs, 1).squeeze(1)
        log_qy_zc = log_probs.gather(1, y.view(-1, 1)).squeeze(1)

        y_onehot = F.one_hot(y, num_classes=model.num_classes).float()
        y_embed = model.y_embedding(y_onehot)
        prior_mu = model.cond_prior_mu(y_embed)
        prior_logvar = model.cond_prior_logvar(y_embed)

        dec = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)

        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)
        log_p_znot = log_normal_diag(
            z_not_c,
            torch.zeros_like(z_not_c),
            torch.zeros_like(z_not_c),
        )

        log_qz_x = log_normal_diag(z, mu, logvar)
        log_p_y = 0.0 if uniform_class_prior else 0.0

        inner = log_p_xz + log_p_zc_y + log_p_znot + log_p_y - log_qy_zc - log_qz_x
        terms.append(inner)

    elbo = torch.stack(terms).mean()

    return elbo, {"elbo": elbo.item()}
