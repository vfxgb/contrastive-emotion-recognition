import torch.nn as nn
from mamba_ssm import Mamba

class ContrastiveMambaEncoder(nn.Module):
    def __init__(self, mamba_args, embed_dim=256, vocab_size=30522):
        super().__init__()
        # Create an embedding layer to map token IDs to embeddings.
        # We assume mamba_args is a dictionary. If it's a ModelArgs instance, adjust accordingly.
        self.embedding = nn.Embedding(vocab_size, mamba_args['d_model'])
        # Unpack the mamba_args dictionary to pass keyword arguments.
        self.mamba = Mamba(**mamba_args)
        # Projection layer to map from d_model to embed_dim.
        self.projection = nn.Linear(mamba_args['d_model'], embed_dim)

    def forward(self, input_ids):
        # Convert token IDs to embeddings. Now shape: (batch, sequence_length, d_model)
        x = self.embedding(input_ids)
        # Process the embedded sequence with Mamba.
        embeddings = self.mamba(x)
        # Mean pooling over the sequence dimension.
        pooled_emb = embeddings.mean(dim=1)
        # Project to the desired embedding space.
        emotion_emb = self.projection(pooled_emb)
        return emotion_emb
    
class ClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_emotions):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_emotions)

    def forward(self, emotion_emb):
        logits = self.classifier(emotion_emb)
        return logits
