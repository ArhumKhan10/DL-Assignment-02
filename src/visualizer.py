"""
Visualization Module

This module handles visualization of:
- Training curves (loss and accuracy)
- Model comparison charts
- Evaluation metrics visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


class TrainingVisualizer:
    """
    Handles visualization of training and evaluation results.
    """
    
    def __init__(self, output_dir: str = 'plots'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_history(self, history: Dict, model_name: str, 
                             save_path: Optional[str] = None):
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{model_name}_training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None):
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            save_path: Path to save plot (optional)
        """
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        # Prepare data
        data = {metric: [metrics_dict[model].get(metric, 0) for model in models] 
                for metric in metric_names}
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            bars = ax.bar(models, data[metric], alpha=0.7, edgecolor='black')
            ax.set_title(metric.upper().replace('_', '-'), fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
        plt.close()
    
    def plot_loss_accuracy_combined(self, histories: Dict[str, Dict],
                                   save_path: Optional[str] = None):
        """
        Plot combined loss and accuracy for multiple models.
        
        Args:
            histories: Dictionary mapping model names to their training histories
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        for model_name, history in histories.items():
            axes[0].plot(history['loss'], label=f'{model_name} - Train', 
                        linewidth=2, linestyle='-')
            axes[0].plot(history['val_loss'], label=f'{model_name} - Val', 
                        linewidth=2, linestyle='--')
        
        axes[0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        for model_name, history in histories.items():
            axes[1].plot(history['accuracy'], label=f'{model_name} - Train', 
                        linewidth=2, linestyle='-')
            axes[1].plot(history['val_accuracy'], label=f'{model_name} - Val', 
                        linewidth=2, linestyle='--')
        
        axes[1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'combined_training_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined training comparison plot saved to {save_path}")
        plt.close()

