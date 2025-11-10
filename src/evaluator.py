"""
Evaluation Module

This module handles model evaluation with:
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Prediction generation
- Qualitative analysis
- Support for both TensorFlow/Keras and PyTorch models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, List, Union
import pandas as pd


class ModelEvaluator:
    """
    Handles evaluation of legal clause similarity models.
    Supports both TensorFlow/Keras and PyTorch models.
    
    Attributes:
        model: Trained model (Keras or PyTorch wrapper)
        model_name: Name identifier for the model
        is_pytorch: Whether the model is a PyTorch model
    """
    
    def __init__(self, model: Union[keras.Model, 'PyTorchModelWrapper'], model_name: str):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model (Keras Model or PyTorchModelWrapper)
            model_name: Name identifier for the model
        """
        self.model = model
        self.model_name = model_name
        # Check if it's a PyTorch model
        self.is_pytorch = hasattr(model, 'predict') and not isinstance(model, keras.Model)
    
    def predict(self, X1: np.ndarray, X2: np.ndarray, 
                threshold: float = 0.5, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            X1: First clause sequences
            X2: Second clause sequences
            threshold: Classification threshold
            
        Returns:
            Tuple of (probability predictions, binary predictions)
        """
        if self.is_pytorch:
            # PyTorch model
            proba_predictions, binary_predictions = self.model.predict(X1, X2, threshold=threshold, batch_size=batch_size)
            return proba_predictions, binary_predictions
        else:
            # TensorFlow/Keras model - use larger batch size for speed
            proba_predictions = self.model.predict([X1, X2], verbose=0, batch_size=batch_size)
            binary_predictions = (proba_predictions >= threshold).astype(int)
            return proba_predictions.flatten(), binary_predictions.flatten()
    
    def evaluate(self, X1: np.ndarray, X2: np.ndarray, y_true: np.ndarray,
                 threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model with multiple metrics.
        
        Args:
            X1: First clause sequences
            X2: Second clause sequences
            y_true: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metric names and values
        """
        # Get predictions
        y_proba, y_pred = self.predict(X1, X2, threshold)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC-AUC (requires probability predictions)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        # PR-AUC (Average Precision)
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics for {self.model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"{'='*60}\n")
    
    def get_qualitative_results(self, X1: np.ndarray, X2: np.ndarray, 
                               y_true: np.ndarray, clause_texts: List[Tuple[str, str]],
                               num_examples: int = 10, batch_size: int = 64) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get qualitative results (correct and incorrect predictions).
        
        Args:
            X1: First clause sequences
            X2: Second clause sequences
            y_true: True labels
            clause_texts: List of (clause1_text, clause2_text) tuples
            num_examples: Number of examples to return
            batch_size: Batch size for predictions (for PyTorch models)
            
        Returns:
            Tuple of (correct_examples_df, incorrect_examples_df)
        """
        print(f"[INFO] Generating predictions for {len(X1)} samples...", flush=True)
        y_proba, y_pred = self.predict(X1, X2, batch_size=batch_size)
        print(f"[OK] Predictions completed", flush=True)
        
        results = []
        for i in range(len(X1)):
            # Truncate long clauses for readability
            clause1_text = clause_texts[i][0]
            clause2_text = clause_texts[i][1]
            
            results.append({
                'clause1': clause1_text[:150] + '...' if len(clause1_text) > 150 else clause1_text,
                'clause2': clause2_text[:150] + '...' if len(clause2_text) > 150 else clause2_text,
                'true_label': 'Similar' if int(y_true[i]) == 1 else 'Different',
                'predicted_label': 'Similar' if int(y_pred[i]) == 1 else 'Different',
                'probability': float(y_proba[i]),
                'correct': int(y_true[i]) == int(y_pred[i])
            })
        
        df = pd.DataFrame(results)
        
        # Get examples
        correct_examples = df[df['correct'] == True].head(num_examples)
        incorrect_examples = df[df['correct'] == False].head(num_examples)
        
        return correct_examples, incorrect_examples

