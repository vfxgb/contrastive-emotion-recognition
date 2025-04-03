import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch import nn
import random

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_emb = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('t-SNE visualization of emotion embeddings')
    plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device

        if features.dim() == 3:
            B, views, D = features.shape
            features = features.view(B * views, D)
            labels = labels.repeat(views)

        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        numerator = (exp_sim * mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -torch.log((numerator[valid] + self.eps) / (denominator[valid] + self.eps))
        return loss.mean()

