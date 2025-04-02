import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from transformers import AutoModel

class ContrastiveMambaEncoder(nn.Module):
    def __init__(self, mamba_args, embed_dim=256):
        """
        Uses BERT to generate embeddings which are then processed by Mamba.
        Args:
            mamba_args (dict): Configuration for our Mamba2 module.
            embed_dim (int): Dimension of the final embedding space.
        """
        super().__init__()
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        # Freeze BERT parameters 
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # BERT outputs 768-dimensional vectors; we need to project them to d_model
        self.proj_bert = nn.Linear(768, mamba_args['d_model'])
        
        # Process the sequence with Mamba2
        self.mamba = Mamba2(**mamba_args)
        
        # Projection layer from d_model to final embed_dim
        self.projection = nn.Linear(mamba_args['d_model'], embed_dim)

    def forward(self, input_ids, attention_mask=None):
        # If no attention mask is provided, create one (assuming padding token is 0)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        # Obtain BERT embeddings (last hidden state)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Project BERT's 768-dim output to mamba's d_model dimension
        x = self.proj_bert(bert_output)  # shape: (batch, seq_len, d_model)
        # Process with Mamba
        embeddings = self.mamba(x)
        # Mean pooling over the sequence dimension
        pooled_emb = embeddings.mean(dim=1)
        # Project to the desired embedding space
        emotion_emb = self.projection(pooled_emb)
        return emotion_emb

class ClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_emotions):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_emotions)

    def forward(self, emotion_emb):
        logits = self.classifier(emotion_emb)
        return logits
