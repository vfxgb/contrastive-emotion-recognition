# BiLSTM Encoders and Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np


class BiLSTM_BERT_Encoder(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, lstm_layers):
        super(BiLSTM_BERT_Encoder, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Optionally freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers if needed
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        if bert_model_name == "bert-large-uncased":
            input_size = 1024
        else:
            raise ValueError("Model does not match input size")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids, attention_mask):
        # get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        return max_pool_output


class BiLSTM_GloVe_Encoder(nn.Module):
    def __init__(self, embedding_matrix_path, hidden_dim, lstm_layers):
        super(BiLSTM_GloVe_Encoder, self).__init__()

        loaded_data = np.load(embedding_matrix_path, allow_pickle=True).item()
        embedding_matrix = loaded_data["embedding_matrix"]
        vocab_size = loaded_data["vocab_size"]
        embed_dim = 300

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids):
        # get glove embeddings
        embeddings = self.embedding(input_ids)

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        return max_pool_output


class BiLSTM_Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rate):
        super(BiLSTM_Classifier, self).__init__()

        # fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Fully connected layers for classification
        # Dropout layer for regularization

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
