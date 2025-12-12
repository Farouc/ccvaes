# loss.py
import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def log_normal_diag(z, mu, logvar):
    """
    Log-density of a diagonal Gaussian distribution:
        N(mu, diag(exp(logvar)))

    Args:
        z (Tensor): Samples, shape (B, D)
        mu (Tensor): Mean, shape (B, D)
        logvar (Tensor): Log-variance, shape (B, D)

    Returns:
        Tensor of shape (B,)
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1,
    )


def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation of log p(x | z).

    Args:
        x (Tensor): Original images
        recon_x (Tensor): Reconstructed images
        recon_type (str): "bce" or "mse"

    Returns:
        Tensor of shape (B,)
    """
    if recon_type == "bce":
        # Binary Cross-Entropy for images in [0, 1]
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        # Mean Squared Error
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)


# =============================================================
# CCVAE SUPERVISED LOSS — Multi-Label (Equation 4)
# =============================================================

def ccvae_loss_supervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    y,          # (B, num_attributes)
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Multi-label version of Equation (4) from the CCVAE paper.

    The supervised ELBO is:
        L(x, y) =
            E_q [
                log p(x | z)
              + sum_i log p(z_ci | y_i)
              + log p(z_not)
              - log q(z | x)
              - sum_i log q(y_i | z_ci)
            ]

    Importance sampling with K samples is used.
    """

    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    # ----------------------------------------------------------
    # 1. Encoder q(z | x)
    # ----------------------------------------------------------
    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    # Importance sampling accumulators
    log_p_xz_list = []
    log_p_zy_list = []
    log_qz_x_list = []
    log_qy_zc_total_list = []
    qy_zc_individual_list = []

    for _ in range(K):
        # ------------------------------------------------------
        # A. Sample z ~ q(z | x)
        # ------------------------------------------------------
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        # Split z into [z_c1, ..., z_cK, z_not_c]
        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        # ------------------------------------------------------
        # B. log p(x | z)
        # ------------------------------------------------------
        dec_in = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)

        # ------------------------------------------------------
        # C. Attribute-wise terms
        # ------------------------------------------------------
        log_p_zc_given_y_sum = 0.0
        log_qy_given_zc_sum = 0.0
        current_k_probs = []

        for i in range(num_attributes):
            # p(z_ci | y_i)
            y_i = y[:, i]
            y_onehot = F.one_hot(
                y_i, num_classes=model.num_classes_list[i]
            ).float()

            embed = F.relu(model.priors_embedding[i](y_onehot))
            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)

            log_p_zc_given_y_sum += log_normal_diag(
                z_c_list[i], p_mu, p_logvar
            )

            # q(y_i | z_ci)
            logits = model.classifiers[i](z_c_list[i])
            log_probs = F.log_softmax(logits, dim=-1)

            log_qy_i = log_probs.gather(
                1, y_i.view(-1, 1)
            ).squeeze(1)

            log_qy_given_zc_sum += log_qy_i
            current_k_probs.append(log_qy_i.exp())

        # ------------------------------------------------------
        # D. Prior for z_not_c
        # ------------------------------------------------------
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)

        log_p_zy = log_p_zc_given_y_sum + log_p_znot
        log_qz_x = log_normal_diag(z, mu, logvar)

        log_p_xz_list.append(log_p_xz)
        log_p_zy_list.append(log_p_zy)
        log_qz_x_list.append(log_qz_x)
        log_qy_zc_total_list.append(log_qy_given_zc_sum)
        qy_zc_individual_list.append(current_k_probs)

    # ----------------------------------------------------------
    # Importance-weighted aggregation
    # ----------------------------------------------------------
    log_q_y_x_total = 0.0
    weights_total = 1.0

    for i in range(num_attributes):
        probs_i_k = torch.stack(
            [k_list[i] for k_list in qy_zc_individual_list], dim=0
        )

        q_yi_x = probs_i_k.mean(dim=0)
        log_q_yi_x = torch.log(q_yi_x + 1e-8)

        log_q_y_x_total += log_q_yi_x

        w_i = probs_i_k / (q_yi_x.unsqueeze(0) + 1e-8)
        weights_total = weights_total * w_i

    log_p_xz = torch.stack(log_p_xz_list, dim=0)
    log_p_zy = torch.stack(log_p_zy_list, dim=0)
    log_qz_x = torch.stack(log_qz_x_list, dim=0)
    log_qy_zc = torch.stack(log_qy_zc_total_list, dim=0)

    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc
    weighted_inner = (weights_total * inner).mean(dim=0)

    L_xy = weighted_inner + log_q_y_x_total
    elbo = L_xy.mean()
    loss = -elbo

    with torch.no_grad():
        stats = {
            "loss": loss.item(),
            "log_p_xz": log_p_xz.mean().item(),
            "log_qy_zc": log_qy_zc.mean().item(),
        }

    return loss, stats


# =============================================================
# CCVAE UNSUPERVISED LOSS — Multi-Label (Equation 5)
# =============================================================

def ccvae_loss_unsupervised_paper(
    model,
    x,
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Multi-label version of Equation (5).

    For each attribute i:
        y_i ~ q(y_i | z_ci)
    """

    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    log_terms_list = []

    for _ in range(K):
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)

        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        log_p_zc_given_y_sum = 0.0
        log_qy_given_zc_sum = 0.0

        for i in range(num_attributes):
            logits = model.classifiers[i](z_c_list[i])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
            log_qy_i = log_probs.gather(
                1, y_sample.view(-1, 1)
            ).squeeze(1)

            log_qy_given_zc_sum += log_qy_i

            y_onehot = F.one_hot(
                y_sample, num_classes=model.num_classes_list[i]
            ).float()

            embed = F.relu(model.priors_embedding[i](y_onehot))
            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)

            log_p_zc_given_y_sum += log_normal_diag(
                z_c_list[i], p_mu, p_logvar
            )

        dec_in = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)

        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)

        log_qz_x = log_normal_diag(z, mu, logvar)
        log_p_zy = log_p_zc_given_y_sum + log_p_znot

        inner = log_p_xz + log_p_zy - log_qy_given_zc_sum - log_qz_x
        log_terms_list.append(inner)

    log_terms = torch.stack(log_terms_list, dim=0)
    L_x = log_terms.mean(dim=0)

    elbo = L_x.mean()
    loss = -elbo

    stats = {
        "loss": loss.item(),
        "elbo": elbo.item(),
    }

    return loss, stats
