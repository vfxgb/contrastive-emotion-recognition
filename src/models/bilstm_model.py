# BiLSTM Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

class BiLSTM(nn.Module):
    """
    BiLSTM model for text classification using BERT embeddings.
    The model consists of a BERT encoder followed by a bidirectional LSTM layer,
    and a series of fully connected layers for classification.

    Args:
        bert_model_name (str): Name of the BERT model to use (e.g., 'bert-base-uncased').
        hidden_dim (int): Hidden dimension size for the LSTM layer.
        num_classes (int): Number of output classes for classification.
        dropout_rate (float): Dropout rate for regularization.
        lstm_layers (int): Number of layers in the LSTM.

    Attributes:
        bert (BertModel): Pre-trained BERT model used for embeddings.
        lstm (nn.LSTM): Bidirectional LSTM layer for sequence modeling.
        global_max_pool (function): Function for performing global max pooling.
        fc1 (nn.Linear): Fully connected layer for classification.
        fc2 (nn.Linear): Second fully connected layer for classification.
        fc3 (nn.Linear): Output fully connected layer for classification.
        dropout (nn.Dropout): Dropout layer for regularization.

    """

    def __init__(
        self, bert_model_name, hidden_dim, num_classes, dropout_rate, lstm_layers
    ):
        super(BiLSTM, self).__init__()

        # Initialize BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Get the input size based on the BERT model name
        if bert_model_name == "bert-large-uncased":
            input_size = 1024
        else:
            # only supports bert-large-uncased for now
            raise ValueError("Model does not match input size")

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Global max pooling layer
        self.global_max_pool = lambda x: torch.max(x, dim=1)[0]

        # fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass throught the model.
        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tensor.
        Returns:
            torch.Tensor: Output logits for classification.

        """
        # get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        # Fully connected layers for classification
        # Dropout layer for regularization
        x = self.dropout(max_pool_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits

class BiLSTM_glove(nn.Module):
    
    def __init__(
        self, embedding_matrix_path, bert_model_name, hidden_dim, num_classes, dropout_rate, lstm_layers
    ):
        super(BiLSTM_glove, self).__init__()

        loaded_data = np.load(embedding_matrix_path, allow_pickle=True).item()

        embedding_matrix = loaded_data["embedding_matrix"]
        vocab_size = loaded_data["vocab_size"]
        embed_dim = 300

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True  # dont freeze embeddings

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Global max pooling layer
        self.global_max_pool = lambda x: torch.max(x, dim=1)[0]

        # fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        """
        Forward pass throught the model.
        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tensor.
        Returns:
            torch.Tensor: Output logits for classification.

        """
        # get BERT embeddings
        embeddings = self.embedding(input_ids)

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        # Fully connected layers for classification
        # Dropout layer for regularization
        x = self.dropout(max_pool_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits
