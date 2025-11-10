"""
Training Pipeline Module

This module handles model training with:
- Training loop management
- Callback configuration
- Model checkpointing
- Training history tracking
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from typing import Dict, Optional, Tuple
import os
import json


class ModelTrainer:
    """
    Handles training of legal clause similarity models.
    
    Attributes:
        model: Keras model to train
        model_name: Name identifier for the model
        history: Training history dictionary
    """
    
    def __init__(self, model: keras.Model, model_name: str, 
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            model_name: Name identifier for the model
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.history = {}
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def compile_model(self, learning_rate: float = 0.001, 
                     optimizer: Optional[str] = None):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('adam', 'rmsprop', etc.) or None for default
        """
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"Model compiled with {optimizer} optimizer")
    
    def get_callbacks(self, monitor: str = 'val_loss', 
                     patience: int = 5,
                     save_best_only: bool = True,
                     target_accuracy: float = 0.995) -> list:
        """
        Get training callbacks.
        
        Args:
            monitor: Metric to monitor
            patience: Patience for early stopping
            save_best_only: Whether to save only best model
            target_accuracy: Target accuracy to stop training (default: 0.99)
            
        Returns:
            List of callbacks
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'{self.model_name}_best.h5'
        )
        
        # Custom callback to stop when accuracy reaches target or plateaus
        class AccuracyStoppingCallback(keras.callbacks.Callback):
            def __init__(self, target_acc=0.99, plateau_patience=3, min_improvement=0.001):
                super().__init__()
                self.target_acc = target_acc
                self.best_acc = 0.0
                self.no_improve_count = 0
                self.plateau_patience = plateau_patience
                self.min_improvement = min_improvement
                self.acc_history = []
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current_acc = logs.get('val_accuracy', logs.get('accuracy', 0.0))
                self.acc_history.append(current_acc)
                
                # Check if we reached target accuracy
                if current_acc >= self.target_acc:
                    print(f"\n[STOPPING] Target accuracy {self.target_acc} reached! "
                          f"Current accuracy: {current_acc:.4f}", flush=True)
                    self.model.stop_training = True
                    return
                
                # Check if accuracy is improving significantly
                improvement = current_acc - self.best_acc
                if improvement >= self.min_improvement:
                    self.best_acc = current_acc
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                
                # Stop if accuracy hasn't improved for plateau_patience epochs
                if self.no_improve_count >= self.plateau_patience and epoch >= 2:
                    print(f"\n[STOPPING] Accuracy plateau detected - no improvement for {self.plateau_patience} epochs. "
                          f"Best accuracy: {self.best_acc:.4f}, Current: {current_acc:.4f}", flush=True)
                    self.model.stop_training = True
                    return
        
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor=monitor,
                save_best_only=save_best_only,
                verbose=1,
                mode='min' if 'loss' in monitor else 'max'
            ),
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            AccuracyStoppingCallback(target_acc=target_accuracy, plateau_patience=3, min_improvement=0.001)
        ]
        
        return callbacks
    
    def train(self, 
              X1_train: tf.Tensor, X2_train: tf.Tensor, y_train: tf.Tensor,
              X1_val: tf.Tensor, X2_val: tf.Tensor, y_val: tf.Tensor,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              verbose: int = 1,
              target_accuracy: float = 0.995) -> Dict:
        """
        Train the model.
        
        Args:
            X1_train: First clause sequences for training
            X2_train: Second clause sequences for training
            y_train: Training labels
            X1_val: First clause sequences for validation
            X2_val: Second clause sequences for validation
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Compile model if not already compiled
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            self.compile_model(learning_rate=learning_rate)
        
        # Get callbacks
        callbacks = self.get_callbacks(target_accuracy=target_accuracy)
        
        # Train model
        print(f"\nTraining {self.model_name}...")
        print(f"Training samples: {len(X1_train)}, Validation samples: {len(X1_val)}")
        
        history = self.model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
        # Load best weights
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'{self.model_name}_best.h5'
        )
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print(f"Loaded best weights from {checkpoint_path}")
        
        return self.history
    
    def save_model(self, save_path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save model (default: checkpoint_dir/model_name.h5)
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, f'{self.model_name}.h5')
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def save_history(self, save_path: Optional[str] = None):
        """
        Save training history to JSON.
        
        Args:
            save_path: Path to save history (default: checkpoint_dir/model_name_history.json)
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_history.json')
        
        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) for v in values]
        
        with open(save_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {save_path}")

