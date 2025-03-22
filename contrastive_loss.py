import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.eye(features.shape[0], device=device).bool()
        mask = mask.masked_fill(logits_mask, 0)

        exp_sim = torch.exp(similarity_matrix) * (~logits_mask)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))

        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob.mean()
        return loss
