import torch
import torch.nn as nn
from transformers import AutoModel
from mamba_ssm import Mamba2

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, embeddings):
        # Compute attention weights over the sequence tokens
        weights = torch.softmax(self.attn(embeddings), dim=1)  # shape: (batch, seq_len, 1)
        pooled_emb = (embeddings * weights).sum(dim=1)
        return pooled_emb

class ContrastiveMambaEncoder(nn.Module):
    def __init__(self, mamba_args, embed_dim=2048):
        """
        Uses BERT to generate embeddings which are then processed by Mamba.
        Args:
            mamba_args (dict): Configuration for the Mamba2 module.
            embed_dim (int): Dimension of the final embedding space.
        """
        super().__init__()
        # Load pretrained BERT model (BERT-large outputs 1024-dimensional vectors)
        self.bert = AutoModel.from_pretrained("bert-large-uncased")
        # Freeze all BERT parameters first
        for param in self.bert.parameters():
            param.requires_grad = False
        # Unfreeze the last two layers for fine-tuning
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Project BERT's 1024-dim output to Mamba's d_model dimension
        self.proj_bert = nn.Linear(1024, mamba_args['d_model'])
        
        # Process the sequence with Mamba2
        self.mamba = Mamba2(**mamba_args)
        
        # Use attention pooling over the sequence dimension 
        self.pooling = AttentionPooling(mamba_args['d_model'])
        
        # Projection layer from mamba's d_model to final embedding space 
        self.projection = nn.Linear(mamba_args['d_model'], embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        # Obtain BERT embeddings (last hidden state)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Project BERT's output to d_model dimension
        x = self.proj_bert(bert_output)
        # Process the sequence with Mamba2
        embeddings = self.mamba(x)
        # Apply attention pooling
        pooled_emb = self.pooling(embeddings)
        pooled_emb = self.dropout(pooled_emb)
        # Final projection to the desired embedding space
        emotion_emb = self.projection(pooled_emb)
        return emotion_emb

class ClassifierHead(nn.Module):
    def __init__(self, embed_dim=2048, num_emotions=7):
        """
        3-layer feedforward neural network with hidden dimensions [2048, 768, num_emotions].
        Softmax can be applied externally (e.g., in the loss function) if needed.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_emotions)
        )

    def forward(self, emotion_emb):
        logits = self.classifier(emotion_emb)
        return logits
