"""
PyTorch Training Pipeline Module

This module handles PyTorch model training with:
- Training loop management
- Validation
- Model checkpointing
- Training history tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from datetime import datetime


class SiameseDataset(Dataset):
    """Dataset class for Siamese network pairs."""
    
    def __init__(self, X1, X2, y):
        """
        Initialize dataset.
        
        Args:
            X1: First clause sequences
            X2: Second clause sequences
            y: Labels
        """
        self.X1 = torch.from_numpy(X1).long()
        self.X2 = torch.from_numpy(X2).long()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


class PyTorchModelTrainer:
    """
    Handles training of PyTorch legal clause similarity models.
    
    Attributes:
        model: PyTorch model to train
        model_name: Name identifier for the model
        device: Device to train on (cpu/cuda)
        history: Training history dictionary
    """
    
    def __init__(self, model: nn.Module, model_name: str,
                 checkpoint_dir: str = 'checkpoints', device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            model_name: Name identifier for the model
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on ('cpu' or 'cuda'), defaults to auto-detect
        """
        self.model = model
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        import time
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        num_batches = len(train_loader)
        print(f"  [TRAIN] Batches: {num_batches}", flush=True)
        
        for batch_idx, (X1, X2, y) in enumerate(train_loader, 1):
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            y = y.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X1, X2).squeeze()
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)

            # Progress log every 200 batches and last batch
            if batch_idx % 200 == 0 or batch_idx == num_batches:
                elapsed = time.time() - start_time
                samples_per_sec = total / max(elapsed, 1e-6)
                avg_loss_so_far = total_loss / batch_idx
                acc_so_far = correct / max(total, 1)
                print(
                    f"    [TRAIN] Batch {batch_idx}/{num_batches} | "
                    f"AvgLoss: {avg_loss_so_far:.4f} | Acc: {acc_so_far:.4f} | "
                    f"Throughput: {samples_per_sec:.1f} samples/s",
                    flush=True,
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        import time
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        num_batches = len(val_loader)
        print(f"  [VAL] Batches: {num_batches}", flush=True)
        
        with torch.no_grad():
            for batch_idx, (X1, X2, y) in enumerate(val_loader, 1):
                X1 = X1.to(self.device)
                X2 = X2.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                outputs = self.model(X1, X2).squeeze()
                
                # Calculate loss
                loss = criterion(outputs, y)
                
                # Statistics
                total_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

                # Progress log every 200 batches and last batch
                if batch_idx % 200 == 0 or batch_idx == num_batches:
                    elapsed = time.time() - start_time
                    samples_per_sec = total / max(elapsed, 1e-6)
                    avg_loss_so_far = total_loss / batch_idx
                    acc_so_far = correct / max(total, 1)
                    print(
                        f"    [VAL] Batch {batch_idx}/{num_batches} | "
                        f"AvgLoss: {avg_loss_so_far:.4f} | Acc: {acc_so_far:.4f} | "
                        f"Throughput: {samples_per_sec:.1f} samples/s",
                        flush=True,
                    )
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, X1_train: np.ndarray, X2_train: np.ndarray, y_train: np.ndarray,
              X1_val: np.ndarray, X2_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
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
            target_accuracy: Target accuracy to stop training
            
        Returns:
            Training history dictionary
        """
        # Create datasets and data loaders
        train_dataset = SiameseDataset(X1_train, X2_train, y_train)
        val_dataset = SiameseDataset(X1_val, X2_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        best_model_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best.pt')
        no_improve_count = 0
        plateau_patience = 3
        min_improvement = 0.0005
        
        import time
        print(f"\nTraining {self.model_name} on {self.device}...")
        print(f"Training samples: {len(X1_train)}, Validation samples: {len(X1_val)}")
        print(f"Target accuracy: {target_accuracy}, Plateau patience: {plateau_patience}\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"[EPOCH {epoch+1}/{epochs}] Starting...", flush=True)
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                  f"Epoch Time: {epoch_time:.1f}s", flush=True)
            
            # Check for target accuracy
            if val_acc >= target_accuracy:
                print(f"\n[STOPPING] Target accuracy {target_accuracy} reached! "
                      f"Current accuracy: {val_acc:.4f}", flush=True)
                self.save_checkpoint(best_model_path)
                break
            
            # Check for improvement
            improvement = val_acc - best_val_acc
            if improvement >= min_improvement:
                best_val_acc = val_acc
                no_improve_count = 0
                # Save best model
                self.save_checkpoint(best_model_path)
                print(f"[INFO] Epoch {epoch+1}: Accuracy improved to {val_acc:.4f} "
                      f"(improvement: {improvement:.4f}) - Checkpoint saved", flush=True)
            elif val_acc > best_val_acc:
                # Even if improvement is small, update best if it's better
                best_val_acc = val_acc
                self.save_checkpoint(best_model_path)
                print(f"[INFO] Epoch {epoch+1}: New best accuracy {val_acc:.4f} - Checkpoint saved", flush=True)
            else:
                no_improve_count += 1
                if improvement < 0:
                    print(f"[WARNING] Epoch {epoch+1}: Accuracy decreased from {best_val_acc:.4f} to {val_acc:.4f} "
                          f"(decrease: {abs(improvement):.4f})", flush=True)
                else:
                    print(f"[INFO] Epoch {epoch+1}: Accuracy not improving significantly "
                          f"(current: {val_acc:.4f}, best: {best_val_acc:.4f}, "
                          f"improvement: {improvement:.4f} < {min_improvement:.4f})", flush=True)
            
            # Check for plateau
            if no_improve_count >= plateau_patience and epoch >= 1:
                print(f"\n[STOPPING] Accuracy not improving for {no_improve_count} epochs. "
                      f"Best accuracy: {best_val_acc:.4f}, Current: {val_acc:.4f}", flush=True)
                # Load best model
                self.load_checkpoint(best_model_path)
                break
        
        # Load best model if not already loaded
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        else:
            # Ensure best model is saved even if training completed all epochs
            if best_val_acc > 0:
                print(f"[INFO] Saving best model checkpoint (val_acc: {best_val_acc:.4f})...", flush=True)
                self.save_checkpoint(best_model_path)
        
        return self.history
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
    
    def save_model(self, save_path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save model (default: checkpoint_dir/model_name.pt)
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, f'{self.model_name}.pt')
        
        # Also save as best checkpoint if it's the attention model
        best_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best.pt')
        
        torch.save(self.model, save_path)
        print(f"Model saved to {save_path}")
        
        # Also save state_dict as best checkpoint for easier loading
        torch.save(self.model.state_dict(), best_path)
        print(f"Best checkpoint saved to {best_path}")
    
    def save_history(self, save_path: Optional[str] = None):
        """
        Save training history to JSON.
        
        Args:
            save_path: Path to save history (default: checkpoint_dir/model_name_history.json)
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_history.json')
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {save_path}")

