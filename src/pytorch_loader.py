"""
PyTorch Model Loader Module

This module handles loading and wrapping PyTorch models for evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os
import pickle
import sys


class PyTorchModelWrapper:
    """
    Wrapper for PyTorch models to make them compatible with the evaluation pipeline.
    """
    
    def __init__(self, model_path: str, model: Optional[nn.Module] = None, device: Optional[str] = None):
        """
        Initialize PyTorch model wrapper.
        
        Args:
            model_path: Path to the PyTorch model file (.pt or .pth)
            model: Optional model architecture (if loading state_dict)
            device: Device to load model on ('cpu' or 'cuda'), defaults to 'cpu'
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.load_model()
    
    def load_model(self):
        """Load the PyTorch model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading PyTorch model from {self.model_path}...", flush=True)
        print(f"Using device: {self.device}", flush=True)
        
        # Try multiple loading strategies (quick-to-slow)
        strategies = [
            self._load_state_dict_quick,   # fast path: weights_only if available
            self._load_torchscript,        # try TorchScript
            self._load_with_state_dict_extraction,  # load and extract state_dict
            self._load_standard            # final fallback
        ]
        
        last_error = None
        for strategy in strategies:
            try:
                strategy()
                print("[OK] PyTorch model loaded successfully!", flush=True)
                return
            except Exception as e:
                last_error = e
                print(f"[INFO] Loading strategy failed: {str(e)[:100]}...", flush=True)
                continue
        
        print(f"[ERROR] Failed to load PyTorch model after all strategies: {last_error}", flush=True)
        raise RuntimeError(f"Could not load PyTorch model: {last_error}")

    def _load_state_dict_quick(self):
        """Fast path: attempt to load weights only and fill provided architecture."""
        try:
            loaded = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Older torch without weights_only
            loaded = torch.load(self.model_path, map_location=self.device)

        # If we got a Module, use it directly
        if isinstance(loaded, nn.Module):
            self.model = loaded
        else:
            # Expect a state_dict-like
            if self.model is None:
                raise ValueError("Need model architecture to load state_dict")
            if isinstance(loaded, dict) and 'state_dict' in loaded:
                state_dict = loaded['state_dict']
            else:
                state_dict = loaded
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def _load_torchscript(self):
        """Try loading as TorchScript model."""
        ts = torch.jit.load(self.model_path, map_location=self.device)
        # Wrap TorchScript in a small adapter that mimics nn.Module forward(x1,x2)
        if self.model is None:
            # Assume the torchscript already takes two inputs
            self.model = ts
        else:
            # If architecture provided, prefer that; TorchScript path not applicable
            raise ValueError("TorchScript path skipped: architecture already provided")
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def _load_standard(self):
        """Standard loading method."""
        loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(loaded, nn.Module):
            self.model = loaded
        elif isinstance(loaded, dict):
            if 'model' in loaded:
                self.model = loaded['model']
            elif self.model is not None:
                self.model.load_state_dict(loaded)
            else:
                raise ValueError("Need model architecture for state_dict")
        else:
            self.model = loaded
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    # Removed custom unpickler path to avoid slow/unsafe unpickling and read-only attribute errors
    
    def _load_with_state_dict_extraction(self):
        """Try to extract state_dict from the saved file."""
        if self.model is None:
            raise ValueError("Need model architecture for state_dict extraction")
        
        # Try loading and extracting just the model weights
        loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Try different ways to extract state_dict
        state_dict = None
        if isinstance(loaded, dict):
            if 'state_dict' in loaded:
                state_dict = loaded['state_dict']
            elif 'model' in loaded and hasattr(loaded['model'], 'state_dict'):
                state_dict = loaded['model'].state_dict()
            else:
                # Assume the dict itself is the state_dict
                state_dict = loaded
        elif isinstance(loaded, nn.Module):
            state_dict = loaded.state_dict()
        elif hasattr(loaded, 'state_dict'):
            state_dict = loaded.state_dict()
        elif hasattr(loaded, 'model') and hasattr(loaded.model, 'state_dict'):
            state_dict = loaded.model.state_dict()
        
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError("Could not extract state_dict")
    
    def predict(self, X1: np.ndarray, X2: np.ndarray, 
                batch_size: int = 64, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from the PyTorch model.
        
        Args:
            X1: First clause sequences (numpy array)
            X2: Second clause sequences (numpy array)
            batch_size: Batch size for inference
            threshold: Classification threshold
            
        Returns:
            Tuple of (probability predictions, binary predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert numpy arrays to torch tensors and clamp values to valid vocab range
        # Clamp to [0, vocab_size-1] to avoid index out of range errors
        vocab_size = None
        if hasattr(self.model, 'embedding'):
            vocab_size = self.model.embedding.num_embeddings
        elif hasattr(self.model, 'vocab_size'):
            vocab_size = self.model.vocab_size
        
        X1_tensor = torch.from_numpy(X1).long().to(self.device)
        X2_tensor = torch.from_numpy(X2).long().to(self.device)
        
        # Clamp indices to valid range if we know vocab_size
        if vocab_size is not None:
            # Check input range before clamping
            input_max = max(X1_tensor.max().item() if X1_tensor.numel() > 0 else 0, 
                          X2_tensor.max().item() if X2_tensor.numel() > 0 else 0)
            if input_max >= vocab_size:
                print(f"[WARNING] Input indices exceed model vocab_size ({vocab_size}). Max index: {input_max}", flush=True)
                print(f"[INFO] Clamping indices to valid range [0, {vocab_size-1}]", flush=True)
            
            # Clamp to [0, vocab_size-1], keeping 0 for padding
            X1_tensor = torch.clamp(X1_tensor, 0, vocab_size - 1)
            X2_tensor = torch.clamp(X2_tensor, 0, vocab_size - 1)
        else:
            print(f"[WARNING] Could not determine model vocab_size, cannot clamp indices", flush=True)
        
        # Get predictions in batches with progress tracking
        predictions = []
        total_batches = (len(X1) + batch_size - 1) // batch_size
        print(f"[INFO] Processing {len(X1)} samples in {total_batches} batches (batch_size={batch_size})...", flush=True)
        
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(X1), batch_size), 1):
                batch_X1 = X1_tensor[i:i+batch_size]
                batch_X2 = X2_tensor[i:i+batch_size]
                
                # Ensure batch is not empty
                if batch_X1.size(0) == 0:
                    continue
                
                # Print progress every 50 batches or on last batch (less frequent for speed)
                if batch_idx % 50 == 0 or batch_idx == total_batches:
                    print(f"[PROGRESS] Processing batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)", flush=True)
                
                # Forward pass - model expects (x1, x2) as two separate arguments
                try:
                    if isinstance(self.model, nn.Module):
                        # Standard Siamese network forward: model(x1, x2)
                        output = self.model(batch_X1, batch_X2)
                    else:
                        raise ValueError(f"Model is not a nn.Module, got {type(self.model)}")
                except Exception as e1:
                    error_msg = str(e1)
                    print(f"[ERROR] Prediction failed at batch {batch_idx}/{total_batches}: {error_msg}", flush=True)
                    print(f"[DEBUG] Batch shapes: X1={batch_X1.shape}, X2={batch_X2.shape}", flush=True)
                    if hasattr(self.model, 'embedding'):
                        print(f"[DEBUG] Embedding vocab_size: {self.model.embedding.num_embeddings}", flush=True)
                        print(f"[DEBUG] X1 range: [{batch_X1.min().item()}, {batch_X1.max().item()}]", flush=True)
                        print(f"[DEBUG] X2 range: [{batch_X2.min().item()}, {batch_X2.max().item()}]", flush=True)
                    raise ValueError(f"Could not get predictions. Error: {error_msg}") from e1
                
                # Handle different output formats
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    # Apply sigmoid if output is not already in [0, 1] range
                    if output.dim() > 1:
                        output = output.squeeze()
                    # Check if sigmoid is needed
                    if output.min() < 0 or output.max() > 1:
                        output = torch.sigmoid(output)
                    predictions.append(output.cpu().numpy())
                else:
                    # Convert to numpy if it's already a numpy array
                    predictions.append(np.array(output))
        
        print(f"[OK] Completed all {total_batches} batches", flush=True)
        
        # Concatenate all predictions
        proba_predictions = np.concatenate(predictions, axis=0)
        
        # Ensure predictions are in [0, 1] range
        proba_predictions = np.clip(proba_predictions, 0, 1)
        
        # Convert to binary predictions
        binary_predictions = (proba_predictions >= threshold).astype(int)
        
        return proba_predictions.flatten(), binary_predictions.flatten()

