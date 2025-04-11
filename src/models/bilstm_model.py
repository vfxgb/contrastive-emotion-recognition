# BiLSTM Encoders and Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np


class BiLSTM_BERT_Encoder(nn.Module):
    """
    BiLSTM encoder using BERT embeddings as input.

    This model uses the contextualized embeddings from pretrained
    bert model, followed by a bidirectional LSTM layer and global
    max pooling to generate a fixed size representation.

    """

    def __init__(self, bert_model_name, hidden_dim, lstm_layers):
        """
        Initializes the BiLSTM_BERT_Encoder.

        Args:
            bert_model_name (str): name of the pretrained bert model.
            hidden_dim (int): hidden size for the LSTM layer.
            lstm_layers (int): number of LSTM layers.
        """
        super(BiLSTM_BERT_Encoder, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # freeze all layers of BERT first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last 2 encoder layers for fine-tuning
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        if bert_model_name == "bert-large-uncased":
            # get input size based on bert model name
            input_size = 1024
        else:
            raise ValueError("Model does not match input size")

        # bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the encoder.

        Args:
            input_ids (Tensor): input token ids
            attention_mask (Tensor): attention mask

        Returns:
            Tensor: Encoded feature representation
        """
        # get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        return max_pool_output


class BiLSTM_GloVe_Encoder(nn.Module):
    """
    BiLSTM encoder using GloVe embeddings as input.

    This model initializes an embedding layer with pretrained GloVe embeddings,
    followed by a bidirectional LSTM layer and global max pooling.

    """

    def __init__(self, embedding_matrix_path, hidden_dim, lstm_layers):
        """
        Initializes the BiLSTM_GloVe_Encoder.

        Args:
            embedding_matrix_path (str): path to the saved GloVe embedding matrix.
            hidden_dim (int): hidden size for the LSTM layer.
            lstm_layers (int): number of LSTM layers.
        """
        super(BiLSTM_GloVe_Encoder, self).__init__()

        # load embedding matrix obtained during preprocessing
        loaded_data = np.load(embedding_matrix_path, allow_pickle=True).item()
        embedding_matrix = loaded_data["embedding_matrix"]
        vocab_size = loaded_data["vocab_size"]
        embed_dim = 300

        # initialise glove embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float)
            )

        # bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids):
        """
        Forward pass of the encoder.

        Args:
            input_ids (Tensor): input token ids

        Returns:
            Tensor: Encoded feature representation
        """
        # get glove embeddings
        embeddings = self.embedding(input_ids)

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embeddings)

        # Apply global max pooling
        max_pool_output = torch.max(lstm_out, dim=1)[0]

        return max_pool_output


class BiLSTM_Classifier(nn.Module):
    """
    A feedforward classifier that takes in the encoded features from the
    BiLSTM encoder and outputs class logits for classification tasks.
    """

    def __init__(self, hidden_dim, num_classes, dropout_rate):
        """
        Initializes the BiLSTM_Classifier.

        Args:
            hidden_dim (int): hidden size from the encoder
            num_classes (int): number of output classes.
            dropout_rate (float): dropout probability for regularization.
        """
        super(BiLSTM_Classifier, self).__init__()

        # fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): input feature tensor.

        Returns:
            Tensor: logits for each class.
        """
        # Fully connected layers for classification
        # Dropout layer for regularization
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
