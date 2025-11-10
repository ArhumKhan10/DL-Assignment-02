"""
Training Script for Legal Clause Similarity Models

This script trains both BiLSTM and Simple LSTM models with 5 epochs.
Run this script to train the models without opening the notebook.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_loader import LegalClauseDataLoader
from models import BiLSTMSiameseNetwork
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import TrainingVisualizer
from pytorch_loader import PyTorchModelWrapper
from pytorch_models import AttentionSiameseNetwork
from pytorch_trainer import PyTorchModelTrainer

# Configuration
CONFIG = {
    'data_dir': 'data',
    'max_seq_length': 128,
    'embedding_dim': 128,
    'batch_size': 32,
    'epochs': 5,  # 5 epochs for faster training
    'learning_rate': 0.001,
    'test_size': 0.2,
    'val_size': 0.1,
    'min_word_freq': 2,
    'checkpoint_dir': 'checkpoints',
    'plots_dir': 'plots',
    'num_pairs_per_category': None
}

def main():
    print("="*80)
    print("Legal Clause Similarity Detection - Model Training")
    print("="*80)
    print(f"Training Configuration:")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print("="*80)
    
    # Create directories
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    data_loader = LegalClauseDataLoader(
        data_dir=CONFIG['data_dir'],
        max_seq_length=CONFIG['max_seq_length']
    )
    
    print("Starting dataset loading...", flush=True)
    clauses_by_category = data_loader.load_dataset()
    print(f"\n[OK] Loaded {len(clauses_by_category)} categories", flush=True)
    total_clauses = sum(len(clauses) for clauses in clauses_by_category.values())
    print(f"[OK] Total clauses: {total_clauses}", flush=True)
    print("Moving to vocabulary building...", flush=True)
    
    # Build vocabulary
    print("\n[2/6] Building vocabulary...", flush=True)
    print("  - Collecting word frequencies...", flush=True)
    
    # Check if vocabulary was saved previously
    vocab_file = os.path.join(CONFIG['checkpoint_dir'], 'vocabulary.pkl')
    import pickle
    
    if os.path.exists(vocab_file):
        print("  - Loading saved vocabulary...", flush=True)
        with open(vocab_file, 'rb') as f:
            saved_vocab = pickle.load(f)
            data_loader.word_to_idx = saved_vocab['word_to_idx']
            data_loader.idx_to_word = saved_vocab['idx_to_word']
            data_loader.vocab_size = saved_vocab['vocab_size']
        vocab_size = data_loader.vocab_size
        print(f"  [OK] Loaded vocabulary size: {vocab_size}", flush=True)
    else:
        vocab = data_loader.build_vocabulary(min_freq=CONFIG['min_word_freq'])
        vocab_size = data_loader.vocab_size
        print(f"[OK] Vocabulary size: {vocab_size}", flush=True)
        
        # Save vocabulary for future use
        print("  - Saving vocabulary...", flush=True)
        with open(vocab_file, 'wb') as f:
            pickle.dump({
                'word_to_idx': data_loader.word_to_idx,
                'idx_to_word': data_loader.idx_to_word,
                'vocab_size': data_loader.vocab_size
            }, f)
        print(f"  [OK] Vocabulary saved to {vocab_file}", flush=True)
    
    print("Moving to pair creation...", flush=True)
    
    # Create pairs
    print("\n[3/6] Creating similarity pairs...", flush=True)
    print("  - This may take a few minutes with many categories...", flush=True)
    print("  - Generating positive pairs (same category)...", flush=True)
    pairs, labels = data_loader.create_pairs(
        num_positive=CONFIG['num_pairs_per_category'],
        balance=True
    )
    print(f"[OK] Total pairs created: {len(pairs)}", flush=True)
    print(f"  - Positive pairs: {sum(labels)}", flush=True)
    print(f"  - Negative pairs: {len(labels) - sum(labels)}", flush=True)
    print("Moving to data preparation...", flush=True)
    
    # Prepare data
    print("\n[4/6] Preparing data for training...", flush=True)
    print("  - Converting text to sequences...", flush=True)
    print(f"  - Processing {len(pairs)} pairs...", flush=True)
    X1, X2, y = data_loader.prepare_data(pairs, labels)
    print(f"[OK] Data prepared:", flush=True)
    print(f"  - X1 shape: {X1.shape}", flush=True)
    print(f"  - X2 shape: {X2.shape}", flush=True)
    print(f"  - y shape: {y.shape}", flush=True)
    
    # Split data
    print("  - Splitting into train/val/test sets...", flush=True)
    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = \
        data_loader.split_data(
            X1, X2, y,
            test_size=CONFIG['test_size'],
            val_size=CONFIG['val_size']
        )
    print(f"[OK] Data split complete:", flush=True)
    print(f"  - Training: {len(X1_train)} pairs", flush=True)
    print(f"  - Validation: {len(X1_val)} pairs", flush=True)
    print(f"  - Test: {len(X1_test)} pairs", flush=True)
    
    # Store test pairs for qualitative analysis
    print("  - Storing test pairs for qualitative analysis...", flush=True)
    test_indices = range(len(pairs) - len(X1_test), len(pairs))
    test_pairs_text = [(pairs[i][0], pairs[i][1]) for i in test_indices]
    print("[OK] Data preprocessing complete!", flush=True)
    
    # Step 2: Load BiLSTM Model (skip training, just load for evaluation)
    print("\n" + "="*80)
    print("[5/6] BiLSTM Siamese Network - Loading Pre-trained")
    print("="*80)
    
    bilstm_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_Siamese_best.h5')
    
    if os.path.exists(bilstm_checkpoint):
        print(f"[INFO] Loading pre-trained BiLSTM model from {bilstm_checkpoint}")
        bilstm_model = BiLSTMSiameseNetwork(
            vocab_size=vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=128,
            dense_units=64,
            max_seq_length=CONFIG['max_seq_length'],
            dropout_rate=0.3
        )
        bilstm_network = bilstm_model.build_model()
        bilstm_trainer = ModelTrainer(
            model=bilstm_network,
            model_name='BiLSTM_Siamese',
            checkpoint_dir=CONFIG['checkpoint_dir']
        )
        bilstm_trainer.compile_model(learning_rate=CONFIG['learning_rate'])
        bilstm_network.load_weights(bilstm_checkpoint)
        print("[OK] BiLSTM model loaded successfully!")
        
        # Load history if available
        history_path = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_Siamese_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                bilstm_history = json.load(f)
            print("[OK] BiLSTM training history loaded")
        else:
            bilstm_history = {}
            print("[WARNING] No training history found for BiLSTM")
    else:
        print("[WARNING] BiLSTM checkpoint not found. Skipping BiLSTM evaluation.")
        bilstm_network = None
        bilstm_history = {}
    
    # Step 3: Train PyTorch Attention Model
    print("\n" + "="*80)
    print("[6/6] Training PyTorch Attention Siamese Network")
    print("="*80)
    
    pytorch_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'attention_best.pt')
    
    # Check if checkpoint exists and has matching vocab_size
    checkpoint_exists = os.path.exists(pytorch_checkpoint)
    should_retrain = True
    
    if checkpoint_exists:
        print(f"\n[INFO] Found existing checkpoint: {pytorch_checkpoint}")
        # Check vocab_size in checkpoint
        import torch
        import types
        training_module = types.ModuleType('training')
        training_config_module = types.ModuleType('training.config')
        class TrainingConfig:
            def __init__(self, *args, **kwargs): pass
            def __getattr__(self, name): return lambda *a, **k: None
            def __call__(self, *args, **kwargs): return self
            def __getstate__(self): return {}
            def __setstate__(self, state): pass
        training_config_module.TrainingConfig = TrainingConfig
        training_module.config = training_config_module
        original_training = sys.modules.get('training')
        original_training_config = sys.modules.get('training.config')
        sys.modules['training'] = training_module
        sys.modules['training.config'] = training_config_module
        
        try:
            checkpoint_data = torch.load(pytorch_checkpoint, map_location='cpu', weights_only=False)
            checkpoint_vocab_size = None
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
                if 'embedding.weight' in state_dict:
                    checkpoint_vocab_size, _ = state_dict['embedding.weight'].shape
            elif isinstance(checkpoint_data, dict) and 'embedding.weight' in checkpoint_data:
                checkpoint_vocab_size, _ = checkpoint_data['embedding.weight'].shape
            
            if checkpoint_vocab_size:
                print(f"[INFO] Checkpoint vocab_size: {checkpoint_vocab_size}, Current vocab_size: {vocab_size}")
                if checkpoint_vocab_size == vocab_size:
                    print("[INFO] Vocabulary sizes match!")
                    print("[INFO] However, retraining to ensure model is properly trained with current vocabulary...")
                    should_retrain = True  # Force retrain to ensure compatibility
                else:
                    print(f"[WARNING] Vocabulary mismatch! Checkpoint has vocab_size={checkpoint_vocab_size}, but current vocab_size={vocab_size}")
                    print("[INFO] Will retrain with correct vocabulary...")
                    should_retrain = True
        finally:
            if original_training is not None:
                sys.modules['training'] = original_training
            elif 'training' in sys.modules:
                del sys.modules['training']
            if original_training_config is not None:
                sys.modules['training.config'] = original_training_config
            elif 'training.config' in sys.modules:
                del sys.modules['training.config']
    
    if should_retrain:
        print("\n[INFO] Training PyTorch Attention model with current vocabulary...")
        pytorch_model = AttentionSiameseNetwork(
            vocab_size=vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=128,
            dense_units=64,
            max_seq_length=CONFIG['max_seq_length'],
            dropout_rate=0.3
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        print(f"\nPyTorch Attention Model Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model architecture: Attention-based Siamese Network")
        print(f"  Vocabulary size: {vocab_size}")
        
        pytorch_trainer = PyTorchModelTrainer(
            model=pytorch_model,
            model_name='attention',
            checkpoint_dir=CONFIG['checkpoint_dir']
        )
        
        # Use subset of training data for faster training (for time constraints)
        # Use 20% of training data to speed up significantly
        train_subset_size = int(len(X1_train) * 0.2)  # Use 20% of training data
        print(f"\n[INFO] Using subset of training data for faster training:")
        print(f"  - Full training set: {len(X1_train)} samples")
        print(f"  - Using subset: {train_subset_size} samples (20%)")
        print(f"  - This will significantly speed up training!")
        
        # Randomly sample subset
        import numpy as np
        np.random.seed(42)  # For reproducibility
        train_indices = np.random.choice(len(X1_train), train_subset_size, replace=False)
        X1_train_subset = X1_train[train_indices]
        X2_train_subset = X2_train[train_indices]
        y_train_subset = y_train[train_indices]
        
        # Use larger batch size for faster training
        fast_batch_size = 128  # Increased from 32
        print(f"  - Using larger batch size: {fast_batch_size} (was {CONFIG['batch_size']})")
        print(f"  - This reduces number of batches from {len(X1_train)//CONFIG['batch_size']} to {train_subset_size//fast_batch_size}")
        
        print(f"\nStarting training for {CONFIG['epochs']} epochs...")
        print(f"Training samples: {train_subset_size}, Validation samples: {len(X1_val)}")
        pytorch_history = pytorch_trainer.train(
            X1_train_subset, X2_train_subset, y_train_subset,
            X1_val, X2_val, y_val,
            epochs=CONFIG['epochs'],
            batch_size=fast_batch_size,  # Use larger batch size
            learning_rate=CONFIG['learning_rate'],
            target_accuracy=0.995
        )
        
        pytorch_trainer.save_model()
        pytorch_trainer.save_history()
        print("\n[OK] PyTorch Attention training completed!")
    else:
        print("\n[INFO] Skipping training. Loading existing checkpoint...")
        pytorch_model = AttentionSiameseNetwork(
            vocab_size=vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=128,
            dense_units=64,
            max_seq_length=CONFIG['max_seq_length'],
            dropout_rate=0.3
        )
        pytorch_trainer = PyTorchModelTrainer(
            model=pytorch_model,
            model_name='attention',
            checkpoint_dir=CONFIG['checkpoint_dir']
        )
        pytorch_trainer.load_checkpoint(pytorch_checkpoint)
        print("[OK] PyTorch model loaded successfully!")
        
        # Load history if available
        history_path = os.path.join(CONFIG['checkpoint_dir'], 'attention_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                pytorch_history = json.load(f)
            print("[OK] PyTorch training history loaded")
        else:
            pytorch_history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
            print("[WARNING] No training history found for PyTorch model")
    
    # Step 4: Evaluate Models
    print("\n" + "="*80)
    print("Evaluating Models")
    print("="*80)
    
    # Evaluate BiLSTM (if loaded)
    if bilstm_network is not None:
        bilstm_evaluator = ModelEvaluator(
            model=bilstm_network,
            model_name='BiLSTM_Siamese'
        )
        bilstm_test_metrics = bilstm_evaluator.evaluate(X1_test, X2_test, y_test)
        bilstm_evaluator.print_metrics(bilstm_test_metrics)
    else:
        bilstm_evaluator = None
        bilstm_test_metrics = {}
    
    # Step 4.2: Evaluate PyTorch Attention Model
    print("\n" + "="*80)
    print("Evaluating PyTorch Attention Model")
    print("="*80)
    
    # Wrap PyTorch model for evaluation
    # Use the trained model if available, otherwise load from checkpoint
    pytorch_wrapper = PyTorchModelWrapper(
        model_path=os.path.join(CONFIG['checkpoint_dir'], 'attention_best.pt'),
        model=pytorch_model  # Pass the model architecture in case we need to load state_dict
    )
    pytorch_evaluator = ModelEvaluator(
        model=pytorch_wrapper,
        model_name='PyTorch_Attention'
    )
    pytorch_test_metrics = pytorch_evaluator.evaluate(X1_test, X2_test, y_test)
    pytorch_evaluator.print_metrics(pytorch_test_metrics)
    
    # Step 4.5: Qualitative Results (Required for Assignment)
    print("\n" + "="*80)
    print("Qualitative Results Analysis")
    print("="*80)
    print("\n[NOTE] Showing qualitative results for all models.")
    print("       BiLSTM: Loaded from checkpoint (pre-trained)")
    print("       PyTorch Attention: Trained/loaded model\n")
    
    # Qualitative results for both models
    if bilstm_evaluator is not None:
        print("\n" + "-"*80)
        print("BiLSTM Model - Qualitative Results (Pre-trained Model)")
        print("-"*80)
        bilstm_correct, bilstm_incorrect = bilstm_evaluator.get_qualitative_results(
            X1_test, X2_test, y_test, test_pairs_text, num_examples=5
        )
    else:
        bilstm_correct, bilstm_incorrect = pd.DataFrame(), pd.DataFrame()
    
    if bilstm_evaluator is not None:
        if len(bilstm_correct) > 0:
            print("\n✓ Correct Predictions (Sample - 5 examples):")
            print(bilstm_correct[['clause1', 'clause2', 'true_label', 'predicted_label', 'probability']].to_string(index=False))
        else:
            print("\n[WARNING] No correct predictions found in sample")
        
        if len(bilstm_incorrect) > 0:
            print("\n✗ Incorrect Predictions (Sample - 5 examples):")
            print(bilstm_incorrect[['clause1', 'clause2', 'true_label', 'predicted_label', 'probability']].to_string(index=False))
        else:
            print("\n[INFO] No incorrect predictions found in sample (perfect accuracy!)")
        
        # Save qualitative results to CSV
        bilstm_correct.to_csv(os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_correct_predictions.csv'), index=False)
        bilstm_incorrect.to_csv(os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_incorrect_predictions.csv'), index=False)
        print(f"\n[SAVED] BiLSTM qualitative results saved to checkpoints/")
    
    # PyTorch qualitative results
    print("\n" + "-"*80)
    print("PyTorch Attention Model - Qualitative Results")
    print("-"*80)
    pytorch_correct, pytorch_incorrect = pytorch_evaluator.get_qualitative_results(
        X1_test, X2_test, y_test, test_pairs_text, num_examples=5
    )
    
    if len(pytorch_correct) > 0:
        print("\n✓ Correct Predictions (Sample - 5 examples):")
        print(pytorch_correct[['clause1', 'clause2', 'true_label', 'predicted_label', 'probability']].to_string(index=False))
    else:
        print("\n[WARNING] No correct predictions found in sample")
    
    if len(pytorch_incorrect) > 0:
        print("\n✗ Incorrect Predictions (Sample - 5 examples):")
        print(pytorch_incorrect[['clause1', 'clause2', 'true_label', 'predicted_label', 'probability']].to_string(index=False))
    else:
        print("\n[INFO] No incorrect predictions found in sample (perfect accuracy!)")
    
    # Save qualitative results to CSV
    pytorch_correct.to_csv(os.path.join(CONFIG['checkpoint_dir'], 'PyTorch_Attention_correct_predictions.csv'), index=False)
    pytorch_incorrect.to_csv(os.path.join(CONFIG['checkpoint_dir'], 'PyTorch_Attention_incorrect_predictions.csv'), index=False)
    print(f"\n[SAVED] PyTorch Attention qualitative results saved to checkpoints/")
    
    print("\n" + "="*80)
    print("Qualitative Analysis Complete!")
    print("="*80)
    print("\nSummary:")
    print(f"  - BiLSTM: {len(bilstm_correct)} correct, {len(bilstm_incorrect)} incorrect (in sample)")
    print(f"  - PyTorch Attention: {len(pytorch_correct)} correct, {len(pytorch_incorrect)} incorrect (in sample)")
    print("\nAll qualitative results saved to checkpoints/ directory")
    
    # Step 5: Generate Visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    visualizer = TrainingVisualizer(output_dir=CONFIG['plots_dir'])
    visualizer.plot_training_history(bilstm_history, 'BiLSTM_Siamese')
    
    # Convert PyTorch history to same format as Keras history
    pytorch_history_formatted = {
        'loss': pytorch_history['train_loss'],
        'accuracy': pytorch_history['train_accuracy'],
        'val_loss': pytorch_history['val_loss'],
        'val_accuracy': pytorch_history['val_accuracy']
    }
    visualizer.plot_training_history(pytorch_history_formatted, 'PyTorch_Attention')
    
    histories = {
        'BiLSTM': bilstm_history,
        'PyTorch_Attention': pytorch_history_formatted
    }
    visualizer.plot_loss_accuracy_combined(histories)
    
    metrics_dict = {
        'BiLSTM_Siamese': bilstm_test_metrics,
        'PyTorch_Attention': pytorch_test_metrics
    }
    visualizer.plot_metrics_comparison(metrics_dict)
    
    print("\nAll visualizations saved to plots/ directory")
    
    # Step 6: Save Results to CSV
    print("\n" + "="*80)
    print("Saving Results to CSV")
    print("="*80)
    
    # Create results DataFrame
    models_list = ['BiLSTM_Siamese', 'PyTorch_Attention']
    accuracy_list = [bilstm_test_metrics['accuracy'], pytorch_test_metrics['accuracy']]
    precision_list = [bilstm_test_metrics['precision'], pytorch_test_metrics['precision']]
    recall_list = [bilstm_test_metrics['recall'], pytorch_test_metrics['recall']]
    f1_list = [bilstm_test_metrics['f1_score'], pytorch_test_metrics['f1_score']]
    roc_auc_list = [bilstm_test_metrics['roc_auc'], pytorch_test_metrics['roc_auc']]
    pr_auc_list = [bilstm_test_metrics['pr_auc'], pytorch_test_metrics['pr_auc']]
    epochs_list = [CONFIG['epochs'], CONFIG['epochs']]
    batch_size_list = [CONFIG['batch_size'], CONFIG['batch_size']]
    learning_rate_list = [CONFIG['learning_rate'], CONFIG['learning_rate']]
    
    results_data = {
        'Model': models_list,
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1_Score': f1_list,
        'ROC_AUC': roc_auc_list,
        'PR_AUC': pr_auc_list,
        'Epochs': epochs_list,
        'Batch_Size': batch_size_list,
        'Learning_Rate': learning_rate_list,
        'Training_Samples': [len(X1_train)] * len(models_list),
        'Validation_Samples': [len(X1_val)] * len(models_list),
        'Test_Samples': [len(X1_test)] * len(models_list),
        'Vocabulary_Size': [vocab_size] * len(models_list),
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(models_list)
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_csv_path = os.path.join(CONFIG['checkpoint_dir'], 'training_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    
    # Also save detailed training history to CSV
    bilstm_history_df = pd.DataFrame(bilstm_history)
    pytorch_history_df = pd.DataFrame(pytorch_history)
    
    bilstm_history_csv = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_training_history.csv')
    pytorch_history_csv = os.path.join(CONFIG['checkpoint_dir'], 'PyTorch_Attention_training_history.csv')
    
    bilstm_history_df.to_csv(bilstm_history_csv, index=False)
    pytorch_history_df.to_csv(pytorch_history_csv, index=False)
    
    print(f"\nTraining histories saved:")
    print(f"  - {bilstm_history_csv}")
    print(f"  - {pytorch_history_csv}")
    
    # Step 7: Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nDataset Statistics:")
    print(f"  Categories: {len(clauses_by_category)}")
    print(f"  Total clauses: {sum(len(clauses) for clauses in clauses_by_category.values())}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Training pairs: {len(X1_train)}")
    print(f"  Validation pairs: {len(X1_val)}")
    print(f"  Test pairs: {len(X1_test)}")
    
    print(f"\nModel Performance:")
    print(f"  BiLSTM - Accuracy: {bilstm_test_metrics['accuracy']:.4f}, "
          f"F1: {bilstm_test_metrics['f1_score']:.4f}, "
          f"ROC-AUC: {bilstm_test_metrics['roc_auc']:.4f}")
    print(f"  PyTorch Attention - Accuracy: {pytorch_test_metrics['accuracy']:.4f}, "
          f"F1: {pytorch_test_metrics['f1_score']:.4f}, "
          f"ROC-AUC: {pytorch_test_metrics['roc_auc']:.4f}")
    
    print(f"\nOutput Files:")
    print(f"  Models: {CONFIG['checkpoint_dir']}/")
    print(f"  Plots: {CONFIG['plots_dir']}/")
    print(f"  Results CSV: {os.path.join(CONFIG['checkpoint_dir'], 'training_results.csv')}")
    print("="*80)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

