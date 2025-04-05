import torch
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch import nn
import random
import re
import string
import spacy 

nlp = spacy.load("en_core_web_sm")

# def visualize_embeddings(embeddings, labels):
#     tsne = TSNE(n_components=2, random_state=42)
#     reduced_emb = tsne.fit_transform(embeddings)
    
#     plt.figure(figsize=(8, 8))
#     scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab10', alpha=0.7)
#     plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.title('t-SNE visualization of emotion embeddings')
#     plt.show()

def set_seed(seed):
    """
    Sets random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fetch_label_mapping(isear = False, crowdflower = False, wassa = False):
    if isear:
        # return label mapping for isear dataset
        return {
            'anger': 0,
            'sadness': 1,
            'disgust': 2,
            'shame': 3,
            'fear': 4,
            'joy': 5,
            'guilt': 6
        }
    elif wassa:
        # return label mapping for wassa dataset
        return {
            'anger': 0,
            'sadness': 1,
            'disgust': 2,
            'fear': 3,
            'joy': 4,
            'surprise': 5
        }

def clean_text(text, extended = True):
    """
    Clean text by removing URLs, mentions, hashtags, extra whitespace,
    and converting to lowercase.
    Args:
        text : the text that is to be cleaned.
        extended : if True, applies lematisation and removes 
            punctuations
    """
    text = re.sub(r'http\S+', '', text)    # Remove URLs
    text = re.sub(r'@\w+', '', text)         # Remove mentions
    text = re.sub(r'#', '', text)            # Remove hashtag symbols
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    if not extended:
        return text.lower()
    
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if token.text not in string.punctuation 
        # and not token.is_stop
    ]
    return " ".join(tokens)

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

