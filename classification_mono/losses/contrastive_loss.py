# losses/contrastive.py
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z,
    labels,
    temperature=0.07,
):
    """
    Supervised contrastive loss (Khosla et al.).
    z: (B, D)
    labels: (B,)
    """

    z = F.normalize(z, dim=1)
    B = z.size(0)

    sim = torch.matmul(z, z.T) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True) - torch.exp(
        torch.tensor(1.0 / temperature, device=z.device)
    )

    log_prob = sim - torch.log(denom + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    return -mean_log_prob_pos.mean()
