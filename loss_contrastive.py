# loss_paper.py
import math
import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def log_normal_diag(z, mu, logvar):
    """
    Log density of a diagonal Gaussian N(mu, diag(exp(logvar))).
    Returns a tensor (B,) if summed over the latent dimension.
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1
    )


def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation of log p(x|z) (up to constant).
    Returns a tensor (B,).
    """
    if recon_type == "bce":
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        # -||x - recon_x||^2, constants ignored
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)


def supervised_contrastive_loss(z_c, labels, temperature=0.07):
    """
    Supervised Contrastive Loss on the latent vector z_c.
    Forces z_c vectors with the same label to be close (cosine similarity),
    and vectors with different labels to be far apart.
   
    z_c: (Batch, Dim)
    labels: (Batch,)
    """
    # Normalize vectors
    z_c = F.normalize(z_c, dim=1)
   
    # Similarity matrix (Batch, Batch)
    similarity_matrix = torch.matmul(z_c, z_c.T)
   
    # Mask for same-class positives (excluding self-loop)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z_c.device)
    mask_no_self = mask - torch.eye(mask.shape[0]).to(z_c.device)
   
    # Exp similarity / temp
    exp_sim = torch.exp(similarity_matrix / temperature)
   
    # Denominator: Sum of all similarities for each anchor (excluding self)
    # Note: A standard SimCLR implementation sums over all negatives.
    # Here we sum over everything except self for stability.
    denominator = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.tensor(1.0/temperature)).to(z_c.device)
   
    # Log prob
    log_prob = similarity_matrix / temperature - torch.log(denominator + 1e-8)
   
    # Compute mean of log-likelihood over positive pairs
    # Avoid division by zero if a class appears only once in batch
    mean_log_prob_pos = (mask_no_self * log_prob).sum(1) / (mask_no_self.sum(1) + 1e-8)
   
    loss = -mean_log_prob_pos.mean()
    return loss


# ==============================================================
#  CCVAE SUPERVISED LOSS — Eq (4) + Contrastive
# ==============================================================

def ccvae_loss_supervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    y,          # (B,) hair_color labels
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
    contrastive_weight=0.0 # Set to >0 to enable contrastive loss
):
    """
    Implements Eq (4) of CCVAE paper for a supervised batch (x, y).
    Optionally adds a Supervised Contrastive Loss on mu_c.
    """

    device = x.device
    B = x.size(0)
    D = model.total_z_dim
    z_c_dim = model.z_c_dim

    # 1. Encoding
    h = model.encoder_conv(x)        # (B, 1024)
    mu = model.fc_mu(h)              # (B, D)
    logvar = model.fc_logvar(h)      # (B, D)

    # 1b. Contrastive Loss Calculation (on the mean mu_c)
    # We use mu instead of z to reduce noise for contrastive learning
    mu_c = mu[:, :z_c_dim]
    loss_contrastive = torch.tensor(0.0, device=device)
    if contrastive_weight > 0:
        loss_contrastive = supervised_contrastive_loss(mu_c, y)

    # p(z_c|y) : conditional prior
    y_onehot = F.one_hot(y, num_classes=model.num_classes).float()
    y_embed = model.y_embedding(y_onehot)
    prior_mu = model.cond_prior_mu(y_embed)
    prior_logvar = model.cond_prior_logvar(y_embed)

    # Accumulate terms over K samples
    log_p_xz_list = []
    log_p_zy_list = []
    log_qz_x_list = []
    log_qy_zc_list = []
    qy_zc_list = []

    for k in range(K):
        # 2. Sampling z ~ q(z|x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)   # (B, D)

        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        # 3. log p(x|z)
        dec_in = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)  # (B,)

        # 4. log p(z|y) = log p(z_c|y) + log p(z_not_c)
        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)
        log_p_zy = log_p_zc_y + log_p_znot

        # 5. log q(z|x)
        log_qz_x = log_normal_diag(z, mu, logvar)

        # 6. q(y|z_c)
        logits = model.classifier(z_c)
        log_probs = F.log_softmax(logits, dim=-1)
        log_qy_zc = log_probs.gather(1, y.view(-1, 1)).squeeze(1)
        qy_zc = log_qy_zc.exp()

        # Store
        log_p_xz_list.append(log_p_xz)
        log_p_zy_list.append(log_p_zy)
        log_qz_x_list.append(log_qz_x)
        log_qy_zc_list.append(log_qy_zc)
        qy_zc_list.append(qy_zc)

    # 7. Aggregate K samples
    log_p_xz = torch.stack(log_p_xz_list, dim=0)    # (K, B)
    log_p_zy = torch.stack(log_p_zy_list, dim=0)    # (K, B)
    log_qz_x = torch.stack(log_qz_x_list, dim=0)    # (K, B)
    log_qy_zc = torch.stack(log_qy_zc_list, dim=0)  # (K, B)
    qy_zc = torch.stack(qy_zc_list, dim=0)          # (K, B)

    # q(y|x) ≈ 1/K ∑_k q(y|z_c^(k))
    q_y_x = qy_zc.mean(dim=0)
    log_q_y_x = torch.log(q_y_x + 1e-8)

    # importance weights w_k = q(y|z_c^k) / q(y|x)
    w = qy_zc / (q_y_x.unsqueeze(0) + 1e-8)

    # log [ p(x|z)p(z|y) / (q(y|z_c) q(z|x)) ]
    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc

    # E_{q(z|x)}[ w * inner ]
    weighted_inner = (w * inner).mean(dim=0)

    # log p(y)
    log_p_y = 0.0 if uniform_class_prior else 0.0

    # L(x,y) = E[...] + log q(y|x) + log p(y)
    L_xy = weighted_inner + log_q_y_x + log_p_y     # (B,)

    elbo = L_xy.mean()
   
    # Final Loss = -ELBO + Contrastive Term
    loss = -elbo + (contrastive_weight * loss_contrastive)

    # Stats
    with torch.no_grad():
        stats = dict(
            elbo=elbo.item(),
            loss=loss.item(),
            loss_contrastive=loss_contrastive.item(),
            log_p_xz=log_p_xz.mean().item(),
            log_p_zy=log_p_zy.mean().item(),
            log_qz_x=log_qz_x.mean().item(),
            log_qy_zc=log_qy_zc.mean().item(),
            log_q_y_x=log_q_y_x.mean().item(),
        )

    return loss, stats


# ==============================================================
#  CCVAE UNSUPERVISED LOSS — Eq (5) (No Contrastive here)
# ==============================================================

def ccvae_loss_unsupervised_paper(
    model,
    x,          # (B, 3, 64, 64)
    K=10,
    recon_type="mse",
    uniform_class_prior=True,
):
    """
    Implements Eq (5) of CCVAE paper for unsupervised batch x.
    Note: Contrastive loss is not applied here as we don't have labels.
    """

    device = x.device
    B = x.size(0)
    D = model.total_z_dim
    z_c_dim = model.z_c_dim

    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    log_terms = []

    for k in range(K):
        # 1. z ~ q(z|x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        z_c = z[:, :z_c_dim]
        z_not_c = z[:, z_c_dim:]

        # 2. y ~ q(y|z_c)
        logits = model.classifier(z_c)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
        log_qy_zc = log_probs.gather(1, y_sample.view(-1, 1)).squeeze(1)

        # 3. prior p(z_c|y_sample)
        y_onehot = F.one_hot(y_sample, num_classes=model.num_classes).float()
        y_embed = model.y_embedding(y_onehot)
        prior_mu = model.cond_prior_mu(y_embed)
        prior_logvar = model.cond_prior_logvar(y_embed)

        # 4. log p(x|z)
        dec_in = model.decoder_input(z).view(-1, 64, 4, 4)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type=recon_type)

        # 5. log p(z|y)
        log_p_zc_y = log_normal_diag(z_c, prior_mu, prior_logvar)
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)
        log_p_zy = log_p_zc_y + log_p_znot

        # 6. log q(z|x)
        log_qz_x = log_normal_diag(z, mu, logvar)

        # 7. Terms
        log_p_y = 0.0 if uniform_class_prior else 0.0
        inner = log_p_xz + log_p_zy + log_p_y - log_qy_zc - log_qz_x
        log_terms.append(inner)

    log_terms = torch.stack(log_terms, dim=0)
    L_x = log_terms.mean(dim=0)

    elbo = L_x.mean()
    loss = -elbo

    stats = dict(
        elbo=elbo.item(),
        loss=loss.item(),
        avg_inner=L_x.mean().item(),
    )

    return loss, stats