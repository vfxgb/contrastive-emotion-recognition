import re
import string
import numpy as np
import spacy
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# BiLSTM Model
class BiGRU(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_classes, 
                 dropout_rate, gru_layers ):
        super(BiGRU, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters 
        for param in self.bert.parameters():
            param.requires_grad = False

        if bert_model_name == "bert-large-uncased":
            input_size = 1024
        else:
            raise ValueError("Model does not match input size")

        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=gru_layers,
            bidirectional=True, 
            batch_first=True
        )

        self.global_max_pool = lambda x: torch.max(x, dim=1)[0]

        # fully connected layers for classification

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state
        
        gru_out, _ = self.gru(embeddings) 
        max_pool_output = torch.max(gru_out, dim=1)[0]  # Global max pooling

        x = self.dropout(max_pool_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits