import torch
import torch.nn.functional as F
from torch import nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        # If features have shape [B, views, D], flatten them to [B * views, D]
        if features.dim() == 3:
            B, views, D = features.shape
            features = features.view(B * views, D)
            # Repeat labels for each view.
            labels = labels.repeat(views)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)  # remove self‑pairs

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        exp_sim = torch.exp(sim)
        numerator = (exp_sim * mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1)

        loss = -torch.log(numerator[valid] / denominator[valid])
        return loss.mean()
