"""
Model Architectures Module

This module contains the neural network architectures for legal clause similarity:
1. BiLSTM-based Siamese Network

All models are trained from scratch without pre-trained embeddings.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


class BiLSTMSiameseNetwork:
    """
    BiLSTM-based Siamese Network for clause similarity.
    
    Architecture:
    - Two identical BiLSTM encoders (shared weights)
    - Dense layers for feature extraction
    - Similarity computation using distance metrics
    - Binary classification output
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 lstm_units: int = 128, dense_units: int = 64,
                 max_seq_length: int = 128, dropout_rate: float = 0.3):
        """
        Initialize BiLSTM Siamese Network.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            dense_units: Number of units in dense layers
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_encoder(self) -> Model:
        """
        Build the shared encoder network.
        
        Returns:
            Encoder model
        """
        input_layer = layers.Input(shape=(self.max_seq_length,), name='encoder_input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_seq_length,
            mask_zero=True,
            name='embedding'
        )(input_layer)
        
        # Bidirectional LSTM
        bilstm = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate),
            name='bilstm'
        )(embedding)
        
        # Global max pooling
        pooled = layers.GlobalMaxPooling1D(name='pooling')(bilstm)
        
        # Dense layers
        dense1 = layers.Dense(self.dense_units, activation='relu', name='dense1')(pooled)
        dropout1 = layers.Dropout(self.dropout_rate, name='dropout1')(dense1)
        dense2 = layers.Dense(self.dense_units // 2, activation='relu', name='dense2')(dropout1)
        
        encoder = Model(inputs=input_layer, outputs=dense2, name='encoder')
        return encoder
    
    def build_model(self) -> Model:
        """
        Build the complete Siamese network.
        
        Returns:
            Complete model for training
        """
        # Shared encoder
        encoder = self.build_encoder()
        
        # Two input branches
        input1 = layers.Input(shape=(self.max_seq_length,), name='clause1')
        input2 = layers.Input(shape=(self.max_seq_length,), name='clause2')
        
        # Encode both clauses
        encoded1 = encoder(input1)
        encoded2 = encoder(input2)
        
        # Compute similarity features
        # Absolute difference
        diff = layers.Subtract(name='difference')([encoded1, encoded2])
        abs_diff = layers.Lambda(lambda x: tf.abs(x), name='abs_difference')(diff)
        
        # Element-wise product
        product = layers.Multiply(name='product')([encoded1, encoded2])
        
        # Concatenate features
        merged = layers.Concatenate(name='merge')([encoded1, encoded2, abs_diff, product])
        
        # Classification layers
        dense1 = layers.Dense(64, activation='relu', name='classifier_dense1')(merged)
        dropout1 = layers.Dropout(self.dropout_rate, name='classifier_dropout1')(dense1)
        dense2 = layers.Dense(32, activation='relu', name='classifier_dense2')(dropout1)
        dropout2 = layers.Dropout(self.dropout_rate, name='classifier_dropout2')(dense2)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='similarity_output')(dropout2)
        
        model = Model(inputs=[input1, input2], outputs=output, name='BiLSTM_Siamese')
        self.model = model
        return model
    
    def get_model(self) -> Model:
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model
