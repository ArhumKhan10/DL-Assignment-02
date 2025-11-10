"""
PyTorch Model Architectures Module

This module contains PyTorch neural network architectures for legal clause similarity:
- Attention-based Siamese Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionSiameseNetwork(nn.Module):
    """
    Attention-based Siamese Network for clause similarity.
    
    Architecture:
    - Two identical encoders with attention mechanism (shared weights)
    - Embedding layer
    - Bidirectional LSTM
    - Attention mechanism
    - Dense layers for feature extraction
    - Similarity computation using distance metrics
    - Binary classification output
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 lstm_units: int = 128, dense_units: int = 64,
                 max_seq_length: int = 128, dropout_rate: float = 0.3):
        """
        Initialize Attention Siamese Network.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units (per direction for BiLSTM)
            dense_units: Number of units in dense layers
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
        """
        super(AttentionSiameseNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_units * 2,  # *2 because bidirectional
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dense layers for encoder output
        self.dense1 = nn.Linear(lstm_units * 2, dense_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        
        # Classification layers
        self.classifier_dense1 = nn.Linear((dense_units // 2) * 4, 64)  # *4 for concat features
        self.classifier_dropout1 = nn.Dropout(dropout_rate)
        self.classifier_dense2 = nn.Linear(64, 32)
        self.classifier_dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(32, 1)
        
    def encode(self, x):
        """
        Encode input sequence using LSTM and attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Encoded representation of shape (batch_size, dense_units // 2)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, lstm_units * 2)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global max pooling
        pooled = torch.max(attn_out, dim=1)[0]  # (batch_size, lstm_units * 2)
        
        # Dense layers
        dense1_out = F.relu(self.dense1(pooled))
        dense1_out = self.dropout1(dense1_out)
        dense2_out = F.relu(self.dense2(dense1_out))
        
        return dense2_out
    
    def forward(self, x1, x2):
        """
        Forward pass for Siamese network.
        
        Args:
            x1: First clause sequences (batch_size, seq_length)
            x2: Second clause sequences (batch_size, seq_length)
            
        Returns:
            Similarity score (batch_size, 1)
        """
        # Encode both clauses
        encoded1 = self.encode(x1)  # (batch_size, dense_units // 2)
        encoded2 = self.encode(x2)  # (batch_size, dense_units // 2)
        
        # Compute similarity features
        # Absolute difference
        abs_diff = torch.abs(encoded1 - encoded2)
        
        # Element-wise product
        product = encoded1 * encoded2
        
        # Concatenate features
        merged = torch.cat([encoded1, encoded2, abs_diff, product], dim=1)
        
        # Classification layers
        classifier_out1 = F.relu(self.classifier_dense1(merged))
        classifier_out1 = self.classifier_dropout1(classifier_out1)
        classifier_out2 = F.relu(self.classifier_dense2(classifier_out1))
        classifier_out2 = self.classifier_dropout2(classifier_out2)
        
        # Output
        output = torch.sigmoid(self.output(classifier_out2))
        
        return output

