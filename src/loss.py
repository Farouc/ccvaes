import math
import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)

# ==============================================================
#  Helpers
# ==============================================================

def log_normal_diag(z, mu, logvar):
    """
    Log density of a diagonal Gaussian N(mu, diag(exp(logvar))).
    Returns a tensor (B,)
    """
    return -0.5 * torch.sum(
        LOG_2PI + logvar + (z - mu) ** 2 / logvar.exp(),
        dim=-1
    )

def log_px_given_z(x, recon_x, recon_type="mse"):
    """
    Approximation of log p(x|z).
    Returns a tensor (B,)
    """
    if recon_type == "bce":
        # BCE for images [0,1]
        per_elem = F.binary_cross_entropy(recon_x, x, reduction="none")
    else:
        # MSE
        per_elem = (recon_x - x) ** 2

    return -per_elem.view(per_elem.size(0), -1).sum(dim=1)

def supervised_contrastive_loss(z_c, labels, temperature=0.07):
    """
    Supervised Contrastive Loss on the latent vector z_c.
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
    denominator = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.tensor(1.0/temperature)).to(z_c.device)
    
    # Log prob
    log_prob = similarity_matrix / temperature - torch.log(denominator + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask_no_self * log_prob).sum(1) / (mask_no_self.sum(1) + 1e-8)
    
    return -mean_log_prob_pos.mean()

# ==============================================================
#  1. CCVAE PAPER LOSS (Monte Carlo - Eq 4 & 5)
#     Handles Single-Label, Multi-Label, and Regression via MC.
# ==============================================================

def ccvae_loss_paper_supervised(
    model, x, y, 
    K=10, 
    recon_type="mse", 
    uniform_class_prior=True,
    contrastive_weight=0.0
):
    """
    Implements Eq (4) using Importance Sampling.
    Adaptable for both Classification (CrossEntropy logic) and Regression (MSE logic).
    """
    
    # Get split dimensions
    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    # 1. Encode q(z|x)
    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    # 1.b Contrastive Loss (Optional) - Calculated on mean mu for stability
    loss_contrastive = torch.tensor(0.0, device=x.device)
    if contrastive_weight > 0:
        # We assume the first attribute is the main one for contrastive tasks usually
        # But we can loop if needed. Here we take the first z_c block.
        mu_c_main = mu[:, :model.z_c_dims[0]]
        # We assume y is (B, NumAttr). We take the first label.
        loss_contrastive = supervised_contrastive_loss(mu_c_main, y[:, 0])

    # Accumulators
    log_p_xz_list = []
    log_p_zy_list = []   
    log_qz_x_list = []
    
    # Lists to store intermediate probs for weight calculation
    qy_zc_individual_list = [] # List of K items, each is list of N tensors

    for k in range(K):
        # --- A. Sampling z ~ q(z|x) ---
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        
        # --- B. Slicing ---
        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        # --- C. log p(x|z) ---
        dec_in = model.decoder_input(z)
        dec_in = dec_in.view(-1, *model.decoder_reshape)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type) # (B,)

        # --- D. Attributes Loop ---
        log_p_zc_given_y_sum = 0
        log_qy_given_zc_sum = 0
        current_k_probs = [] 

        for i in range(num_attributes):
            y_i = y[:, i] # (B,)
            task = model.task_types[i]
            
            # -- 1. Prior p(z_c | y) --
            if task == 'classification':
                y_in = F.one_hot(y_i.long(), num_classes=model.num_classes_or_dim[i]).float()
            else: # regression
                y_in = y_i.view(-1, 1).float()

            if task == 'classification':
                embed = F.relu(model.priors_embedding[i](y_in))
            else:
                 # Regression prior net has its own relu inside
                embed = model.priors_embedding[i](y_in)

            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)
            
            log_p_zc_given_y_sum += log_normal_diag(z_c_list[i], p_mu, p_logvar)

            # -- 2. Prediction q(y | z_c) --
            # Classification -> LogSoftmax
            # Regression -> Gaussian Log Likelihood (equivalent to -MSE)
            pred = model.heads[i](z_c_list[i])

            if task == 'classification':
                log_probs = F.log_softmax(pred, dim=-1)
                log_qy_i = log_probs.gather(1, y_i.long().view(-1, 1)).squeeze(1) # (B,)
                prob_i = log_qy_i.exp()
            else:
                # For regression, we treat prediction as Gaussian mean with fixed var=1
                # log N(y; pred, 1) = -0.5 * (y - pred)^2 + C
                # This integrates MSE into the probabilistic framework
                mse_element = (y_i.float() - pred.squeeze(1)) ** 2
                log_qy_i = -0.5 * mse_element
                # For importance weights w, we need the density "prob"
                prob_i = torch.exp(log_qy_i)

            log_qy_given_zc_sum += log_qy_i
            current_k_probs.append(prob_i)

        # -- Prior p(z_not) --
        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)

        # Totals
        log_p_zy = log_p_zc_given_y_sum + log_p_znot
        log_qz_x = log_normal_diag(z, mu, logvar)

        # Store
        log_p_xz_list.append(log_p_xz)
        log_p_zy_list.append(log_p_zy)
        log_qz_x_list.append(log_qz_x)
        qy_zc_individual_list.append(current_k_probs)

    # ----------------------------------------------------------
    # Aggregation & Importance Weights (Eq 4)
    # ----------------------------------------------------------
    
    # Calculate q(y|x) aggregated
    log_q_y_x_total = 0
    weights_total = 1.0 
    
    for i in range(num_attributes):
        # stack K samples for attribute i
        probs_i_k = torch.stack([k_list[i] for k_list in qy_zc_individual_list], dim=0) # (K, B)
        
        # mean over K -> q(yi|x)
        q_yi_x = probs_i_k.mean(dim=0) # (B,)
        log_q_yi_x = torch.log(q_yi_x + 1e-8)
        log_q_y_x_total += log_q_yi_x
        
        # weights w_i = q(yi|z_c) / q(yi|x)
        w_i = probs_i_k / (q_yi_x.unsqueeze(0) + 1e-8)
        weights_total = weights_total * w_i

    # Stack Lists to Tensors
    log_p_xz = torch.stack(log_p_xz_list, dim=0)
    log_p_zy = torch.stack(log_p_zy_list, dim=0)
    log_qz_x = torch.stack(log_qz_x_list, dim=0)
    
    # We re-calculate total log_qy_zc from individual terms for consistency
    # (Doing it inside the loop was hard to stack)
    log_qy_zc_k = torch.stack([
        sum([torch.log(k_list[i] + 1e-8) for i in range(num_attributes)]) 
        for k_list in qy_zc_individual_list
    ], dim=0)

    # Inner term: log [ p(x|z)p(z|y) / (q(y|z_c) q(z|x)) ]
    inner = log_p_xz + log_p_zy - log_qz_x - log_qy_zc_k

    # Expectation E_{q(z|x)} [ w * inner ]
    weighted_inner = (weights_total * inner).mean(dim=0) # (B,)

    # L(x,y) = Weighted_Inner + log q(y|x)
    L_xy = weighted_inner + log_q_y_x_total

    elbo = L_xy.mean()
    loss = -elbo + (contrastive_weight * loss_contrastive)

    stats = {
        "loss": loss.item(),
        "elbo": elbo.item(),
        "contrastive": loss_contrastive.item()
    }
    return loss, stats

# ==============================================================
#  2. ANALYTIC / SIMPLE LOSS (Recon + Beta*KL + Gamma*Pred)
#     Optimized for Regression and stability.
#     Doesn't use Monte Carlo K-loop. Uses closed form KL.
# ==============================================================

def ccvae_loss_simple_analytic(
    model, x, y, 
    beta=1.0, 
    gamma=1.0, 
    recon_type="mse"
):
    """
    Standard VAE Loss with Auxiliary Task.
    L = Recon + Beta * KL_Total + Gamma * Pred_Loss
    Preferred for simple regression tasks (UTKFace).
    """
    B = x.size(0)
    
    # 1. Forward Pass (returns mu/logvar directly)
    # Note: Model forward returns recon_x, mu, logvar, preds[], priors_mu[], priors_logvar[]
    recon_x, mu, logvar, preds, priors_mu_list, priors_logvar_list = model(x, y)

    # 2. Reconstruction Loss
    if recon_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 3. KL Divergence (Analytic)
    # Split mu/logvar
    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    mu_parts = torch.split(mu, split_sizes, dim=1)
    logvar_parts = torch.split(logvar, split_sizes, dim=1)
    
    mu_c_list = mu_parts[:-1]
    logvar_c_list = logvar_parts[:-1]
    
    mu_not = mu_parts[-1]
    logvar_not = logvar_parts[-1]

    # A. KL for z_not_c (Standard Normal Prior)
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_not_c = -0.5 * torch.sum(1 + logvar_not - mu_not.pow(2) - logvar_not.exp())

    # B. KL for z_c attributes (Conditional Prior)
    kld_c_total = 0
    pred_loss_total = 0

    for i in range(model.num_attributes):
        # KL( q(z_ci|x) || p(z_ci|yi) )
        mu_q, logvar_q = mu_c_list[i], logvar_c_list[i]
        mu_p, logvar_p = priors_mu_list[i], priors_logvar_list[i] # From model forward
        
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        
        # Analytic KL between two Gaussians
        # 0.5 * sum( log(vp/vq) + (vq + (mq-mp)^2)/vp - 1 )
        kl_element = logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1.0
        kld_c_total += 0.5 * torch.sum(kl_element)

        # 4. Prediction Loss (Gamma term)
        y_i = y[:, i]
        y_pred = preds[i]
        
        if model.task_types[i] == 'classification':
            # Cross Entropy
            pred_loss_total += F.cross_entropy(y_pred, y_i.long(), reduction='sum')
        else:
            # MSE for Regression
            pred_loss_total += F.mse_loss(y_pred.view(-1), y_i.float(), reduction='sum')

    # Total KL
    kld_loss = kld_not_c + kld_c_total

    # Final Weighted Sum
    total_loss = recon_loss + (beta * kld_loss) + (gamma * pred_loss_total)
    
    # Average per batch for logging consistency
    avg_loss = total_loss / B

    stats = {
        "loss": avg_loss.item(),
        "recon": recon_loss.item() / B,
        "kld": kld_loss.item() / B,
        "pred": pred_loss_total.item() / B
    }

    return avg_loss, stats

# ==============================================================
#  3. UNSUPERVISED LOSS (Eq 5)
# ==============================================================

def ccvae_loss_paper_unsupervised(
    model, x, 
    K=10, 
    recon_type="mse"
):
    """
    Implements Eq (5) for Unsupervised data.
    Similar to supervised but samples y from q(y|z_c).
    """
    split_sizes = model.z_c_dims + [model.z_not_c_dim]
    num_attributes = model.num_attributes

    h = model.encoder_conv(x)
    mu = model.fc_mu(h)
    logvar = model.fc_logvar(h)

    log_terms_list = []

    for k in range(K):
        # Sample z
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        z_parts = torch.split(z, split_sizes, dim=1)
        z_c_list = z_parts[:-1]
        z_not_c = z_parts[-1]

        # Accumulators
        log_p_zc_given_y_sum = 0
        log_qy_given_zc_sum = 0
        
        for i in range(num_attributes):
            # 1. Predict q(y|z)
            pred = model.heads[i](z_c_list[i])
            
            if model.task_types[i] == 'classification':
                log_probs = F.log_softmax(pred, dim=-1)
                probs = log_probs.exp()
                # Sample y
                y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
                log_qy_i = log_probs.gather(1, y_sample.view(-1, 1)).squeeze(1)
                
                # Prep for Prior
                y_in = F.one_hot(y_sample, num_classes=model.num_classes_or_dim[i]).float()
            else:
                # For unsupervised regression, we can't easily "sample" a continuous value 
                # from a single prediction without a variance head. 
                # We usually take the mean prediction.
                y_sample = pred
                # Fake log prob for regression sample (assuming close to mean)
                log_qy_i = torch.zeros(x.size(0), device=x.device) 
                
                y_in = y_sample # (B, 1)

            # 2. Prior p(z|y_sample)
            if model.task_types[i] == 'classification':
                embed = F.relu(model.priors_embedding[i](y_in))
            else:
                embed = model.priors_embedding[i](y_in)

            p_mu = model.priors_mu[i](embed)
            p_logvar = model.priors_logvar[i](embed)
            
            log_p_zc_given_y_sum += log_normal_diag(z_c_list[i], p_mu, p_logvar)
            log_qy_given_zc_sum += log_qy_i

        # Rest of terms
        dec_in = model.decoder_input(z).view(-1, *model.decoder_reshape)
        recon_x = model.decoder_conv(dec_in)
        log_p_xz = log_px_given_z(x, recon_x, recon_type)

        zero_mu = torch.zeros_like(z_not_c)
        zero_logvar = torch.zeros_like(z_not_c)
        log_p_znot = log_normal_diag(z_not_c, zero_mu, zero_logvar)
        log_qz_x = log_normal_diag(z, mu, logvar)

        log_p_zy = log_p_zc_given_y_sum + log_p_znot
        
        # Inner term
        inner = log_p_xz + log_p_zy - log_qy_given_zc_sum - log_qz_x
        log_terms_list.append(inner)

    # Average over K
    log_terms = torch.stack(log_terms_list, dim=0)
    L_x = log_terms.mean(dim=0)
    
    elbo = L_x.mean()
    loss = -elbo

    stats = {"loss": loss.item(), "elbo": elbo.item()}
    return loss, stats