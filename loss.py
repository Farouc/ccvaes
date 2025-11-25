# loss_paper.py

import math
import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def log_normal_diag(z, mu, logvar):
    """
    Log densité d'une gaussienne diagonale N(mu, diag(exp(logvar))).
    Retourne un tenseur (B,) si on somme sur la dimension latente.
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1
    )


def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation de log p(x|z) (up to constant).
    Retourne un tenseur (B,).
    """
    if recon_type == "bce":
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        # -||x - recon_x||^2, les constantes sont ignorées
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)


# ==============================================================
#  CCVAE SUPERVISED LOSS — équation (4)
# ==============================================================

def ccvae_loss_supervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    y,          # (B,) entiers hair_color
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Implémente l'équation (4) du papier CCVAE pour un batch supervisé (x, y).

    L_CCvae(x,y) = E_{q(z|x)}[
        ( q(y|z_c) / q(y|x) ) *
        log ( p(x|z)p(z|y) / ( q(y|z_c) q(z|x) ) )
    ] + log q(y|x) + log p(y).

    On approxime par Monte Carlo avec K échantillons z ~ q(z|x).

    Retourne:
        loss (scalaire à minimiser = -ELBO),
        stats (dict) pour logging.
    """

    device = x.device
    B = x.size(0)
    D = model.total_z_dim
    z_c_dim = model.z_c_dim

    # ----------------------------------------------------------
    # 1. q(z|x) = N(mu, diag(exp(logvar)))
    # ----------------------------------------------------------
    h = model.encoder_conv(x)        # (B, 1024)
    mu = model.fc_mu(h)              # (B, D)
    logvar = model.fc_logvar(h)      # (B, D)

    # p(z_c|y) : prior conditionnel
    y_onehot = F.one_hot(y, num_classes=model.num_classes).float()
    y_embed = model.y_embedding(y_onehot)           # (B, d_embed)
    prior_mu = model.cond_prior_mu(y_embed)         # (B, z_c_dim)
    prior_logvar = model.cond_prior_logvar(y_embed) # (B, z_c_dim)

    # Accumulation des termes sur K samples
    log_p_xz_list = []
    log_p_zy_list = []
    log_qz_x_list = []
    log_qy_zc_list = []
    qy_zc_list = []

    for k in range(K):
        # -----------------------------
        # 2. échantillonnage z ~ q(z|x)
        # -----------------------------
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)   # (B, D)

        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        # -----------------------------
        # 3. log p(x|z)
        # -----------------------------
        dec_in = model.decoder_input(z)          # (B, 64*4*4)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)     # (B, 3, 64, 64)

        log_p_xz = log_px_given_z(x, recon_x, recon_type)  # (B,)

        # -----------------------------
        # 4. log p(z|y) = log p(z_c|y) + log p(z_not_c)
        # -----------------------------
        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)  # (B,)

        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)  # (B,)

        log_p_zy = log_p_zc_y + log_p_znot  # (B,)

        # -----------------------------
        # 5. log q(z|x)
        # -----------------------------
        log_qz_x = log_normal_diag(z, mu, logvar)  # (B,)

        # -----------------------------
        # 6. q(y|z_c)
        # -----------------------------
        logits = model.classifier(z_c)             # (B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        log_qy_zc = log_probs.gather(1, y.view(-1, 1)).squeeze(1)  # (B,)
        qy_zc = log_qy_zc.exp()                    # (B,)

        # Store
        log_p_xz_list.append(log_p_xz)
        log_p_zy_list.append(log_p_zy)
        log_qz_x_list.append(log_qz_x)
        log_qy_zc_list.append(log_qy_zc)
        qy_zc_list.append(qy_zc)

    # ----------------------------------------------------------
    # 7. Agrégation des K échantillons
    # ----------------------------------------------------------
    log_p_xz = torch.stack(log_p_xz_list, dim=0)    # (K, B)
    log_p_zy = torch.stack(log_p_zy_list, dim=0)    # (K, B)
    log_qz_x = torch.stack(log_qz_x_list, dim=0)    # (K, B)
    log_qy_zc = torch.stack(log_qy_zc_list, dim=0)  # (K, B)
    qy_zc = torch.stack(qy_zc_list, dim=0)          # (K, B)

    # q(y|x) ≈ 1/K ∑_k q(y|z_c^(k))
    q_y_x = qy_zc.mean(dim=0)                       # (B,)
    log_q_y_x = torch.log(q_y_x + 1e-8)             # (B,)

    # importance weights w_k = q(y|z_c^k) / q(y|x)
    w = qy_zc / (q_y_x.unsqueeze(0) + 1e-8)         # (K, B)

    # log [ p(x|z)p(z|y) / (q(y|z_c) q(z|x)) ]
    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc  # (K, B)

    # E_{q(z|x)}[ w * inner ]
    weighted_inner = (w * inner).mean(dim=0)        # (B,)

    # log p(y) : si uniforme, constante → peut être ignorée
    if uniform_class_prior:
        log_p_y = 0.0
    else:
        # prior quelconque possible ici
        log_p_y = 0.0

    # L(x,y) = E[...] + log q(y|x) + log p(y)
    L_xy = weighted_inner + log_q_y_x + log_p_y     # (B,)

    elbo = L_xy.mean()
    loss = -elbo

    # Stats pour logging
    with torch.no_grad():
        stats = dict(
            elbo=elbo.item(),
            loss=loss.item(),
            log_p_xz=log_p_xz.mean().item(),
            log_p_zy=log_p_zy.mean().item(),
            log_qz_x=log_qz_x.mean().item(),
            log_qy_zc=log_qy_zc.mean().item(),
            log_q_y_x=log_q_y_x.mean().item(),
        )

    return loss, stats


# ==============================================================
#  CCVAE UNSUPERVISED LOSS — équation (5)
# ==============================================================

def ccvae_loss_unsupervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Implémente l'équation (5) du papier CCVAE pour un batch non-supervisé x.

    L_CCvae(x) = E_{q(z|x) q(y|z_c)}[
        log ( p(x|z)p(z|y)p(y) / (q(y|z_c) q(z|x)) )
    ].

    Approximation Monte Carlo avec K échantillons de (z, y).
    """

    device = x.device
    B = x.size(0)
    D = model.total_z_dim
    z_c_dim = model.z_c_dim

    # q(z|x)
    h = model.encoder_conv(x)
    mu = model.fc_mu(h)          # (B, D)
    logvar = model.fc_logvar(h)  # (B, D)

    log_terms = []

    for k in range(K):
        # -----------------------------
        # 1. z ~ q(z|x)
        # -----------------------------
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)   # (B, D)

        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        # -----------------------------
        # 2. y ~ q(y|z_c)
        # -----------------------------
        logits = model.classifier(z_c)         # (B, C)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        log_qy_zc = log_probs.gather(1, y_sample.view(-1, 1)).squeeze(1)  # (B,)

        # -----------------------------
        # 3. prior conditionnel p(z_c|y_sample)
        # -----------------------------
        y_onehot = F.one_hot(y_sample, num_classes=model.num_classes).float()
        y_embed = model.y_embedding(y_onehot)
        prior_mu = model.cond_prior_mu(y_embed)         # (B, z_c_dim)
        prior_logvar = model.cond_prior_logvar(y_embed) # (B, z_c_dim)

        # -----------------------------
        # 4. log p(x|z)
        # -----------------------------
        dec_in = model.decoder_input(z)
        dec_in = dec_in.view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)

        log_p_xz = log_px_given_z(x, recon_x, recon_type=recon_type)  # (B,)

        # -----------------------------
        # 5. log p(z|y) = log p(z_c|y) + log p(z_not_c)
        # -----------------------------
        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)  # (B,)
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)  # (B,)
        log_p_zy = log_p_zc_y + log_p_znot

        # -----------------------------
        # 6. log q(z|x)
        # -----------------------------
        log_qz_x = log_normal_diag(z, mu, logvar)  # (B,)

        # -----------------------------
        # 7. log [ p(x|z)p(z|y)p(y) / (q(y|z_c) q(z|x)) ]
        # -----------------------------
        if uniform_class_prior:
            log_p_y = 0.0
        else:
            log_p_y = 0.0  # à adapter si prior non uniforme

        inner = log_p_xz + log_p_zy + log_p_y - log_qy_zc - log_qz_x  # (B,)
        log_terms.append(inner)

    log_terms = torch.stack(log_terms, dim=0)  # (K, B)
    L_x = log_terms.mean(dim=0)                # (B,)

    elbo = L_x.mean()
    loss = -elbo

    stats = dict(
        elbo=elbo.item(),
        loss=loss.item(),
        avg_inner=L_x.mean().item(),
    )

    return loss, stats
