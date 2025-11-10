"""
Generate PDF Report for Legal Clause Similarity Assignment

This script creates a comprehensive PDF report with:
- Network details (architecture, parameters, training settings)
- Dataset splits
- Training graphs
- Performance measures
- Performance comparison
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

# Add src to path
sys.path.append('src')

# Configuration
CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'results_dir': 'qualitative_results',
    'output_pdf': 'A2-CS452_Report.pdf'  # Update with your FastID
}

def load_training_histories():
    """Load training histories for both models."""
    histories = {}
    
    # BiLSTM History
    bilstm_json = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_Siamese_history.json')
    if os.path.exists(bilstm_json):
        with open(bilstm_json, 'r') as f:
            histories['BiLSTM'] = json.load(f)
    
    # PyTorch Attention History
    attention_json = os.path.join(CONFIG['checkpoint_dir'], 'attention_history.json')
    if os.path.exists(attention_json):
        with open(attention_json, 'r') as f:
            histories['PyTorch_Attention'] = json.load(f)
    
    return histories

def load_performance_metrics():
    """Load performance metrics from summary CSV."""
    summary_file = os.path.join(CONFIG['results_dir'], 'qualitative_analysis_summary.csv')
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        return df
    return None

def get_training_settings():
    """Get training settings for both models."""
    return {
        'BiLSTM_Siamese': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 5,
            'optimizer': 'Adam',
            'loss': 'binary_crossentropy',
            'early_stopping': 'Yes (target: 0.995)',
            'dropout': 0.3
        },
        'PyTorch_Attention': {
            'batch_size': 128,  # Used larger batch for faster training
            'learning_rate': 0.001,
            'epochs': 5,
            'optimizer': 'Adam',
            'loss': 'BCELoss',
            'early_stopping': 'Yes (target: 0.995, plateau patience: 3)',
            'dropout': 0.3,
            'training_data_subset': '20% (for faster training)'
        }
    }

def create_pdf_report():
    """Create comprehensive PDF report."""
    print("="*80)
    print("Generating PDF Report")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    histories = load_training_histories()
    metrics_df = load_performance_metrics()
    training_settings = get_training_settings()
    
    # Get dataset splits (from report or calculate)
    dataset_splits = {
        'training': 275947,
        'validation': 39421,
        'test': 78842,
        'total': 275947 + 39421 + 78842
    }
    
    # Network details
    network_details = {
        'BiLSTM_Siamese': {
            'architecture': 'TensorFlow/Keras BiLSTM Siamese Network',
            'parameters': 7765729,
            'embedding_dim': 128,
            'lstm_units': 128,
            'dense_units': 64,
            'max_seq_length': 128,
            'vocab_size': 58388
        },
        'PyTorch_Attention': {
            'architecture': 'PyTorch Attention-based Siamese Network',
            'parameters': 8029921,
            'embedding_dim': 128,
            'lstm_units': 128,
            'dense_units': 64,
            'max_seq_length': 128,
            'vocab_size': 58388,
            'attention_heads': 8
        }
    }
    
    print("[OK] Data loaded")
    
    # Create PDF
    print("\n[2/5] Creating PDF document...")
    pdf_path = CONFIG['output_pdf']
    
    # Use matplotlib to create PDF with multiple pages
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(11, 8.5))  # Letter size
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = "Legal Clause Similarity Detection\nDeep Learning Assignment No. 02\n"
        title_text += "NLP Models for Semantic Similarity\n\n"
        title_text += f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}\n"
        title_text += "\n" + "="*70 + "\n\n"
        title_text += "This report contains:\n"
        title_text += "• Network details (architecture, parameters, training settings)\n"
        title_text += "• Dataset splits\n"
        title_text += "• Training graphs\n"
        title_text += "• Performance measures\n"
        title_text += "• Performance comparison of NLP architectures\n"
        
        ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
                fontsize=12, ha='center', va='center', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Network Details
        print("[3/5] Adding network details...")
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        network_text = "NETWORK DETAILS\n" + "="*70 + "\n\n"
        
        for model_name, details in network_details.items():
            settings = training_settings.get(model_name, {})
            network_text += f"{model_name}\n"
            network_text += "-" * 70 + "\n"
            network_text += f"Architecture: {details['architecture']}\n"
            network_text += f"Total Parameters: {details['parameters']:,}\n"
            network_text += f"Embedding Dimension: {details['embedding_dim']}\n"
            network_text += f"LSTM Units: {details['lstm_units']}\n"
            network_text += f"Dense Units: {details['dense_units']}\n"
            network_text += f"Max Sequence Length: {details['max_seq_length']}\n"
            network_text += f"Vocabulary Size: {details['vocab_size']:,}\n"
            if 'attention_heads' in details:
                network_text += f"Attention Heads: {details['attention_heads']}\n"
            network_text += "\nTraining Settings:\n"
            network_text += f"  Batch Size: {settings.get('batch_size', 'N/A')}\n"
            network_text += f"  Learning Rate: {settings.get('learning_rate', 'N/A')}\n"
            network_text += f"  Epochs: {settings.get('epochs', 'N/A')}\n"
            network_text += f"  Optimizer: {settings.get('optimizer', 'N/A')}\n"
            network_text += f"  Loss Function: {settings.get('loss', 'N/A')}\n"
            network_text += f"  Early Stopping: {settings.get('early_stopping', 'N/A')}\n"
            network_text += f"  Dropout Rate: {settings.get('dropout', 'N/A')}\n"
            if 'training_data_subset' in settings:
                network_text += f"  Training Data: {settings['training_data_subset']}\n"
            network_text += "\n" + "="*70 + "\n\n"
        
        network_text += "\nRationale for Baseline Selection:\n"
        network_text += "-" * 70 + "\n"
        network_text += "• BiLSTM: Captures bidirectional context, essential for understanding\n"
        network_text += "  legal clause semantics from both directions.\n"
        network_text += "• Attention Mechanism: Allows the model to focus on relevant parts\n"
        network_text += "  of clauses when determining similarity, improving interpretability.\n"
        network_text += "• Siamese Architecture: Enables direct comparison of clause pairs\n"
        network_text += "  using shared encoders, ideal for similarity tasks.\n"
        
        ax.text(0.05, 0.95, network_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='left', family='monospace',
                wrap=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Dataset Splits
        print("[4/5] Adding dataset splits...")
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        splits_text = "DATASET SPLITS\n" + "="*70 + "\n\n"
        splits_text += f"Total Pairs: {dataset_splits['total']:,}\n\n"
        splits_text += f"Training Set: {dataset_splits['training']:,} pairs ({100*dataset_splits['training']/dataset_splits['total']:.1f}%)\n"
        splits_text += f"Validation Set: {dataset_splits['validation']:,} pairs ({100*dataset_splits['validation']/dataset_splits['total']:.1f}%)\n"
        splits_text += f"Test Set: {dataset_splits['test']:,} pairs ({100*dataset_splits['test']/dataset_splits['total']:.1f}%)\n\n"
        splits_text += "="*70 + "\n\n"
        splits_text += "Dataset Statistics:\n"
        splits_text += f"• Total Categories: 395\n"
        splits_text += f"• Total Clauses: 150,881\n"
        splits_text += f"• Vocabulary Size: 58,388 words\n"
        splits_text += f"• Positive Pairs: 197,105 (50%)\n"
        splits_text += f"• Negative Pairs: 197,105 (50%)\n"
        splits_text += f"• Balanced Dataset: Yes\n"
        
        # Create a pie chart for dataset splits
        ax2 = fig.add_subplot(212)
        sizes = [dataset_splits['training'], dataset_splits['validation'], dataset_splits['test']]
        labels = ['Training', 'Validation', 'Test']
        colors = ['#66b3ff', '#99ff99', '#ffcc99']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Dataset Split Distribution', fontsize=12, fontweight='bold')
        
        ax.text(0.05, 0.95, splits_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='left', family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4-5: Training Graphs
        print("[5/5] Adding training graphs...")
        
        # BiLSTM Training History
        if 'BiLSTM' in histories:
            bilstm_hist = histories['BiLSTM']
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            
            epochs = range(1, len(bilstm_hist.get('val_accuracy', [])) + 1)
            
            # Loss plot
            axes[0].plot(epochs, bilstm_hist.get('loss', []), 'b-', label='Train Loss', linewidth=2)
            axes[0].plot(epochs, bilstm_hist.get('val_loss', []), 'r-', label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Loss', fontsize=11)
            axes[0].set_title('BiLSTM Siamese Network - Training and Validation Loss', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[1].plot(epochs, bilstm_hist.get('accuracy', []), 'b-', label='Train Accuracy', linewidth=2)
            axes[1].plot(epochs, bilstm_hist.get('val_accuracy', []), 'r-', label='Val Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Accuracy', fontsize=11)
            axes[1].set_title('BiLSTM Siamese Network - Training and Validation Accuracy', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # PyTorch Attention Training History
        if 'PyTorch_Attention' in histories:
            pt_hist = histories['PyTorch_Attention']
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            
            epochs = range(1, len(pt_hist.get('val_accuracy', [])) + 1)
            
            # Loss plot
            axes[0].plot(epochs, pt_hist.get('train_loss', []), 'b-', label='Train Loss', linewidth=2)
            axes[0].plot(epochs, pt_hist.get('val_loss', []), 'r-', label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Loss', fontsize=11)
            axes[0].set_title('PyTorch Attention Network - Training and Validation Loss', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[1].plot(epochs, pt_hist.get('train_accuracy', []), 'b-', label='Train Accuracy', linewidth=2)
            axes[1].plot(epochs, pt_hist.get('val_accuracy', []), 'r-', label='Val Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Accuracy', fontsize=11)
            axes[1].set_title('PyTorch Attention Network - Training and Validation Accuracy', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Add existing training graph images if available
        graph_files = [
            ('BiLSTM_Siamese_training_history.png', 'BiLSTM Training History'),
            ('PyTorch_Attention_training_history.png', 'PyTorch Attention Training History'),
            ('combined_training_comparison.png', 'Combined Training Comparison'),
            ('metrics_comparison.png', 'Metrics Comparison')
        ]
        
        for graph_file, title in graph_files:
            graph_path = os.path.join(CONFIG['results_dir'], graph_file)
            if os.path.exists(graph_path):
                try:
                    img = Image.open(graph_path)
                    fig = plt.figure(figsize=(11, 8.5))
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"  [WARNING] Could not add {graph_file}: {e}")
        
        # Page: Performance Measures
        print("[6/7] Adding performance measures...")
        if metrics_df is not None:
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            perf_text = "PERFORMANCE MEASURES\n" + "="*70 + "\n\n"
            perf_text += "Evaluation Metrics on Test Set:\n\n"
            
            # Create formatted table
            for idx, row in metrics_df.iterrows():
                perf_text += f"{row['Model']}\n"
                perf_text += "-" * 70 + "\n"
                perf_text += f"  Accuracy:  {float(row['Accuracy']):.4f}\n"
                perf_text += f"  Precision: {float(row['Precision']):.4f}\n"
                perf_text += f"  Recall:    {float(row['Recall']):.4f}\n"
                perf_text += f"  F1-Score:  {float(row['F1-Score']):.4f}\n"
                perf_text += f"  ROC-AUC:   {float(row['ROC-AUC']):.4f}\n"
                perf_text += f"  Correct Predictions (Sample): {int(row['Correct (Sample)'])}\n"
                perf_text += f"  Incorrect Predictions (Sample): {int(row['Incorrect (Sample)'])}\n\n"
            
            perf_text += "\n" + "="*70 + "\n\n"
            perf_text += "Metric Rationale:\n"
            perf_text += "-" * 70 + "\n"
            perf_text += "• Accuracy: Overall correctness (suitable for balanced dataset)\n"
            perf_text += "• Precision: Important when false positives are costly\n"
            perf_text += "• Recall: Critical for finding all similar clauses\n"
            perf_text += "• F1-Score: Balanced metric combining precision and recall\n"
            perf_text += "• ROC-AUC: Measures ranking ability across thresholds\n"
            perf_text += "• PR-AUC: Better for imbalanced datasets\n\n"
            perf_text += "For production systems, F1-Score and ROC-AUC are most suitable\n"
            perf_text += "as they provide balanced performance assessment.\n"
            
            ax.text(0.05, 0.95, perf_text, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page: Performance Comparison
        print("[7/7] Adding performance comparison...")
        if metrics_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            
            models = metrics_df['Model'].tolist()
            accuracy = [float(x) for x in metrics_df['Accuracy']]
            f1_score = [float(x) for x in metrics_df['F1-Score']]
            roc_auc = [float(x) for x in metrics_df['ROC-AUC']]
            precision = [float(x) for x in metrics_df['Precision']]
            
            # Accuracy comparison
            axes[0, 0].bar(models, accuracy, color=['#66b3ff', '#99ff99'])
            axes[0, 0].set_ylabel('Accuracy', fontsize=10)
            axes[0, 0].set_title('Accuracy Comparison', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylim([0.9, 1.0])
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(accuracy):
                axes[0, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # F1-Score comparison
            axes[0, 1].bar(models, f1_score, color=['#66b3ff', '#99ff99'])
            axes[0, 1].set_ylabel('F1-Score', fontsize=10)
            axes[0, 1].set_title('F1-Score Comparison', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylim([0.9, 1.0])
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(f1_score):
                axes[0, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # ROC-AUC comparison
            axes[1, 0].bar(models, roc_auc, color=['#66b3ff', '#99ff99'])
            axes[1, 0].set_ylabel('ROC-AUC', fontsize=10)
            axes[1, 0].set_title('ROC-AUC Comparison', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylim([0.95, 1.0])
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(roc_auc):
                axes[1, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Precision comparison
            axes[1, 1].bar(models, precision, color=['#66b3ff', '#99ff99'])
            axes[1, 1].set_ylabel('Precision', fontsize=10)
            axes[1, 1].set_title('Precision Comparison', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylim([0.85, 1.0])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(precision):
                axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Performance comparison table with timing
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            comp_text = "PERFORMANCE COMPARISON\n" + "="*70 + "\n\n"
            comp_text += "Quantitative Comparison:\n\n"
            comp_text += f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Precision':<12}\n"
            comp_text += "-" * 70 + "\n"
            for idx, row in metrics_df.iterrows():
                comp_text += f"{row['Model']:<25} {float(row['Accuracy']):<12.4f} {float(row['F1-Score']):<12.4f} "
                comp_text += f"{float(row['ROC-AUC']):<12.4f} {float(row['Precision']):<12.4f}\n"
            
            comp_text += "\n" + "="*70 + "\n\n"
            comp_text += "Key Observations:\n"
            comp_text += "-" * 70 + "\n"
            comp_text += "• PyTorch Attention model achieves higher accuracy (97.14% vs 94.80%)\n"
            comp_text += "• Both models show excellent performance with F1-Scores > 0.94\n"
            comp_text += "• Attention mechanism provides better precision (95.00% vs 90.47%)\n"
            comp_text += "• BiLSTM has slightly higher recall (99.88% vs 99.50%)\n"
            comp_text += "• Both models demonstrate strong ROC-AUC scores (> 0.98)\n\n"
            
            comp_text += "Strengths and Weaknesses:\n"
            comp_text += "-" * 70 + "\n"
            comp_text += "BiLSTM Siamese:\n"
            comp_text += "  Strengths: High recall, faster training, simpler architecture\n"
            comp_text += "  Weaknesses: Lower precision, may over-predict similarity\n\n"
            comp_text += "PyTorch Attention:\n"
            comp_text += "  Strengths: Higher accuracy, better precision, attention interpretability\n"
            comp_text += "  Weaknesses: More complex, slower training, more parameters\n"
            
            ax.text(0.05, 0.95, comp_text, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page: Qualitative Analysis - Correct/Incorrect Examples
        print("[8/8] Adding qualitative analysis (correct/incorrect examples)...")
        
        # Load qualitative results
        for model_name in ['BiLSTM_Siamese', 'PyTorch_Attention']:
            correct_file = os.path.join(CONFIG['results_dir'], f'{model_name}_correct_predictions.csv')
            incorrect_file = os.path.join(CONFIG['results_dir'], f'{model_name}_incorrect_predictions.csv')
            
            if os.path.exists(correct_file) and os.path.exists(incorrect_file):
                correct_df = pd.read_csv(correct_file).head(5)  # Top 5 examples
                incorrect_df = pd.read_csv(incorrect_file).head(5)  # Top 5 examples
                
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                qual_text = f"QUALITATIVE ANALYSIS - {model_name}\n" + "="*70 + "\n\n"
                
                qual_text += "Correct Predictions (Sample):\n"
                qual_text += "-" * 70 + "\n"
                for idx, row in correct_df.iterrows():
                    qual_text += f"\nExample {idx + 1}:\n"
                    clause1 = str(row['clause1'])[:100] + "..." if len(str(row['clause1'])) > 100 else str(row['clause1'])
                    clause2 = str(row['clause2'])[:100] + "..." if len(str(row['clause2'])) > 100 else str(row['clause2'])
                    qual_text += f"  Clause 1: {clause1}\n"
                    qual_text += f"  Clause 2: {clause2}\n"
                    qual_text += f"  True Label: {row['true_label']}, Predicted: {row['predicted_label']}\n"
                    qual_text += f"  Probability: {float(row['probability']):.4f}\n"
                
                qual_text += "\n\n" + "="*70 + "\n\n"
                qual_text += "Incorrect Predictions (Sample):\n"
                qual_text += "-" * 70 + "\n"
                for idx, row in incorrect_df.iterrows():
                    qual_text += f"\nExample {idx + 1}:\n"
                    clause1 = str(row['clause1'])[:100] + "..." if len(str(row['clause1'])) > 100 else str(row['clause1'])
                    clause2 = str(row['clause2'])[:100] + "..." if len(str(row['clause2'])) > 100 else str(row['clause2'])
                    qual_text += f"  Clause 1: {clause1}\n"
                    qual_text += f"  Clause 2: {clause2}\n"
                    qual_text += f"  True Label: {row['true_label']}, Predicted: {row['predicted_label']}\n"
                    qual_text += f"  Probability: {float(row['probability']):.4f}\n"
                    qual_text += f"  Error Type: {'False Positive' if row['true_label'] == 'Different' else 'False Negative'}\n"
                
                ax.text(0.05, 0.95, qual_text, transform=ax.transAxes,
                        fontsize=8, va='top', ha='left', family='monospace')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    print(f"\n[OK] PDF report generated: {pdf_path}")
    print("="*80)

if __name__ == '__main__':
    create_pdf_report()

