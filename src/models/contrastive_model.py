import torch
import torch.nn as nn
from mamba import Mamba, ModelArgs

class ContrastiveMambaModel(nn.Module):
    def __init__(self, mamba_args, num_emotions, embed_dim=256):
        super().__init__()
        self.mamba = Mamba(mamba_args)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(mamba_args.d_model, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_emotions)

    def forward(self, input_ids):
        embeddings = self.mamba(input_ids)
        pooled_emb = embeddings.mean(dim=1)
        emotion_emb = self.projection(pooled_emb)
        logits = self.classifier(emotion_emb)
        return logits, emotion_emb
