import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)  # Normalize embeddings

        batch_size = features.size(0)
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Mask-out self-contrast cases
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask  # Only true positives (not self)

        # Compute log_prob
        exp_logits = torch.exp(sim_matrix) * logits_mask  # remove diagonal
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss
