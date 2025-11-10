"""
Qualitative Analysis Script for Legal Clause Similarity Models

This script performs qualitative analysis on pre-trained models:
- Loads saved models from checkpoints
- Evaluates on test data
- Generates qualitative results (correct/incorrect predictions)
- Saves results to CSV files

Run this script independently after models have been trained.
"""

import os
import sys
import json
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_loader import LegalClauseDataLoader
from models import BiLSTMSiameseNetwork
from evaluator import ModelEvaluator
from pytorch_loader import PyTorchModelWrapper
from pytorch_models import AttentionSiameseNetwork
from visualizer import TrainingVisualizer
import time

# Configuration
CONFIG = {
    'data_dir': 'data',
    'max_seq_length': 128,
    'embedding_dim': 128,
    'test_size': 0.2,
    'val_size': 0.1,
    'min_word_freq': 2,
    'checkpoint_dir': 'checkpoints',
    'results_dir': 'qualitative_results',  # New directory for qualitative results
    'num_examples': 10,  # Number of examples to show and save
    'max_eval_samples': 5000,  # Limit test samples for faster evaluation
    'skip_full_results': True,  # Skip generating full results to save time
    'batch_size_pytorch': 256,  # Larger batch size for PyTorch
    'batch_size_keras': 128  # Larger batch size for Keras
}

def load_model(model_name: str, vocab_size: int, checkpoint_dir: str):
    """
    Load a pre-trained model from checkpoint.
    
    Args:
        model_name: Name of the model ('BiLSTM_Siamese' or 'PyTorch_Attention')
        vocab_size: Vocabulary size
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Loaded model or None if checkpoint not found
    """
    if model_name == 'BiLSTM_Siamese':
        checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.h5')
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            return None

        print(f"\n[INFO] Loading {model_name} from {checkpoint_path}")
        model = BiLSTMSiameseNetwork(
            vocab_size=vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=128,
            dense_units=64,
            max_seq_length=CONFIG['max_seq_length'],
            dropout_rate=0.3
        )
        network = model.build_model()
        network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        network.load_weights(checkpoint_path)
        print(f"[OK] {model_name} loaded successfully!")
        return network

    if model_name == 'PyTorch_Attention':
        # Try sanitized checkpoint first, fallback to original
        sanitized_path = os.path.join(checkpoint_dir, 'attention_sanitized.pt')
        checkpoint_path = os.path.join(checkpoint_dir, 'attention_best.pt')
        
        # Prefer sanitized checkpoint if it exists
        if os.path.exists(sanitized_path):
            checkpoint_path = sanitized_path
            print(f"\n[INFO] Using sanitized checkpoint: {checkpoint_path}")
        elif not os.path.exists(checkpoint_path):
            print(f"[WARNING] PyTorch checkpoint not found: {checkpoint_path}")
            print(f"[INFO] Try running: python sanitize_pytorch_checkpoint.py --checkpoint {checkpoint_path} --output {sanitized_path}")
            return None
        else:
            print(f"\n[INFO] Using original checkpoint (consider sanitizing for faster loading): {checkpoint_path}")

        print(f"\n[INFO] Loading {model_name} from {checkpoint_path}")
        
        # First, check the checkpoint's vocab_size
        import torch
        import types
        # Monkey-patch for loading
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
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_vocab_size = None
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
                if 'embedding.weight' in state_dict:
                    checkpoint_vocab_size, _ = state_dict['embedding.weight'].shape
            elif isinstance(checkpoint_data, dict) and 'embedding.weight' in checkpoint_data:
                checkpoint_vocab_size, _ = checkpoint_data['embedding.weight'].shape
            
            if checkpoint_vocab_size:
                print(f"[INFO] Checkpoint was trained with vocab_size={checkpoint_vocab_size}", flush=True)
                print(f"[INFO] Current data vocab_size={vocab_size}", flush=True)
                if checkpoint_vocab_size != vocab_size:
                    print(f"\n[ERROR] ========================================", flush=True)
                    print(f"[ERROR] VOCABULARY MISMATCH DETECTED!", flush=True)
                    print(f"[ERROR] ========================================", flush=True)
                    print(f"[ERROR] The model checkpoint was trained with vocab_size={checkpoint_vocab_size}", flush=True)
                    print(f"[ERROR] But your current vocabulary has vocab_size={vocab_size}", flush=True)
                    print(f"[ERROR] This causes random predictions (~0.5 accuracy) because:", flush=True)
                    print(f"[ERROR]   - Word-to-index mappings don't match", flush=True)
                    print(f"[ERROR]   - Model embeddings were trained for different words", flush=True)
                    print(f"[ERROR]", flush=True)
                    print(f"[ERROR] SOLUTION: Retrain the model with your vocabulary:", flush=True)
                    print(f"[ERROR]   python train_models.py", flush=True)
                    print(f"[ERROR] ========================================\n", flush=True)
                    return None
                else:
                    print(f"[OK] Vocabulary sizes match!", flush=True)
        finally:
            if original_training is not None:
                sys.modules['training'] = original_training
            elif 'training' in sys.modules:
                del sys.modules['training']
            if original_training_config is not None:
                sys.modules['training.config'] = original_training_config
            elif 'training.config' in sys.modules:
                del sys.modules['training.config']
        
        # Always create the model architecture first to handle missing module dependencies
        print(f"[INFO] Creating model architecture with vocab_size={vocab_size}...")
        attention_model = AttentionSiameseNetwork(
            vocab_size=vocab_size,
            embedding_dim=CONFIG['embedding_dim'],
            lstm_units=128,
            dense_units=64,
            max_seq_length=CONFIG['max_seq_length'],
            dropout_rate=0.3
        )
        
        # Try loading with architecture (handles both full models and state_dicts)
        try:
            wrapper = PyTorchModelWrapper(model_path=checkpoint_path, model=attention_model)
            print(f"[OK] {model_name} loaded successfully!")
            return wrapper
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch model: {e}")
            if checkpoint_path != sanitized_path:
                print(f"[INFO] Try sanitizing the checkpoint first:")
                print(f"  python sanitize_pytorch_checkpoint.py --checkpoint {checkpoint_path} --output {sanitized_path}")
            raise

    print(f"[ERROR] Unknown model name: {model_name}")
    return None

def main():
    print("="*80)
    print("Qualitative Analysis for Legal Clause Similarity Models")
    print("="*80)
    print("\nThis script analyzes pre-trained models and generates qualitative results.")
    print("Make sure models have been trained and saved in checkpoints/ directory.\n")
    
    # Create results directory
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    data_loader = LegalClauseDataLoader(
        data_dir=CONFIG['data_dir'],
        max_seq_length=CONFIG['max_seq_length']
    )
    
    print("  - Loading dataset...", flush=True)
    clauses_by_category = data_loader.load_dataset()
    print(f"  [OK] Loaded {len(clauses_by_category)} categories", flush=True)
    
    print("  - Loading vocabulary...", flush=True)
    vocab_file = os.path.join(CONFIG['checkpoint_dir'], 'vocabulary.pkl')
    import pickle as pkl
    
    if os.path.exists(vocab_file):
        print("  - Loading saved vocabulary from training...", flush=True)
        with open(vocab_file, 'rb') as f:
            saved_vocab = pkl.load(f)
            data_loader.word_to_idx = saved_vocab['word_to_idx']
            data_loader.idx_to_word = saved_vocab['idx_to_word']
            data_loader.vocab_size = saved_vocab['vocab_size']
        vocab_size = data_loader.vocab_size
        print(f"  [OK] Loaded vocabulary size: {vocab_size}", flush=True)
        print(f"  [INFO] Using same vocabulary as training to ensure consistency", flush=True)
    else:
        print("  [WARNING] No saved vocabulary found! Building new vocabulary...", flush=True)
        print("  [WARNING] This may cause vocabulary mismatch with trained models!", flush=True)
        vocab = data_loader.build_vocabulary(min_freq=CONFIG['min_word_freq'])
        vocab_size = data_loader.vocab_size
        print(f"  [OK] Vocabulary size: {vocab_size}", flush=True)
    
    # Check for cached pairs
    pairs_cache_file = os.path.join(CONFIG['checkpoint_dir'], 'pairs_cache.npz')
    pairs_text_cache_file = os.path.join(CONFIG['checkpoint_dir'], 'pairs_text_cache.pkl')
    
    if os.path.exists(pairs_cache_file) and os.path.exists(pairs_text_cache_file):
        print("  - Loading cached pairs...", flush=True)
        try:
            cache_data = np.load(pairs_cache_file)
            X1 = cache_data['X1']
            X2 = cache_data['X2']
            y = cache_data['y']
            with open(pairs_text_cache_file, 'rb') as f:
                pairs_text_data = pkl.load(f)
                pairs = pairs_text_data['pairs']
                labels = pairs_text_data['labels']
            print(f"  [OK] Loaded {len(pairs)} pairs from cache", flush=True)
        except Exception as e:
            print(f"  [WARNING] Failed to load cache: {e}, regenerating...", flush=True)
            print("  - Creating pairs...", flush=True)
            pairs, labels = data_loader.create_pairs(balance=True)
            print(f"  [OK] Total pairs: {len(pairs)}", flush=True)
            print("  - Preparing data...", flush=True)
            X1, X2, y = data_loader.prepare_data(pairs, labels)
            # Save cache
            print("  - Saving pairs cache...", flush=True)
            np.savez(pairs_cache_file, X1=X1, X2=X2, y=y)
            with open(pairs_text_cache_file, 'wb') as f:
                pkl.dump({'pairs': pairs, 'labels': labels}, f)
            print(f"  [OK] Saved pairs cache to {pairs_cache_file}", flush=True)
    else:
        print("  - Creating pairs...", flush=True)
        pairs, labels = data_loader.create_pairs(balance=True)
        print(f"  [OK] Total pairs: {len(pairs)}", flush=True)
        print("  - Preparing data...", flush=True)
        X1, X2, y = data_loader.prepare_data(pairs, labels)
        # Save cache
        print("  - Saving pairs cache...", flush=True)
        np.savez(pairs_cache_file, X1=X1, X2=X2, y=y)
        with open(pairs_text_cache_file, 'wb') as f:
            pkl.dump({'pairs': pairs, 'labels': labels}, f)
        print(f"  [OK] Saved pairs cache to {pairs_cache_file}", flush=True)
    
    print("  - Splitting data...", flush=True)
    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = \
        data_loader.split_data(
            X1, X2, y,
            test_size=CONFIG['test_size'],
            val_size=CONFIG['val_size']
        )
    print(f"  [OK] Test set: {len(X1_test)} pairs", flush=True)
    
    # Store test pairs for qualitative analysis
    # Test set is the last portion after splitting
    test_start_idx = len(pairs) - len(X1_test)
    test_pairs_text = [(pairs[i][0], pairs[i][1]) for i in range(test_start_idx, len(pairs))]
    
    print("\n[OK] Data preprocessing complete!")
    
    # Step 2: Load models
    print("\n[2/4] Loading pre-trained models...")
    
    models_to_analyze = []
    model_details = {}
    
    # Try to load BiLSTM
    bilstm_model = load_model('BiLSTM_Siamese', vocab_size, CONFIG['checkpoint_dir'])
    if bilstm_model is not None:
        models_to_analyze.append(('BiLSTM_Siamese', bilstm_model))
        # Network details for BiLSTM
        try:
            bilstm_params = bilstm_model.count_params()
        except Exception:
            bilstm_params = None
        model_details['BiLSTM_Siamese'] = {
            'architecture': 'TensorFlow/Keras BiLSTM Siamese with dense head',
            'parameters': bilstm_params,
            'embedding_dim': CONFIG['embedding_dim'],
            'max_seq_length': CONFIG['max_seq_length']
        }
    
    # Try to load PyTorch Attention
    pytorch_model = load_model('PyTorch_Attention', vocab_size, CONFIG['checkpoint_dir'])
    if pytorch_model is not None:
        models_to_analyze.append(('PyTorch_Attention', pytorch_model))
        # Network details for PyTorch model
        try:
            # Wrapper exposes .model (nn.Module)
            pt_params = sum(p.numel() for p in getattr(pytorch_model, 'model').parameters())
        except Exception:
            pt_params = None
        model_details['PyTorch_Attention'] = {
            'architecture': 'PyTorch Attention-based Siamese (BiLSTM + MultiHeadAttention)',
            'parameters': pt_params,
            'embedding_dim': CONFIG['embedding_dim'],
            'max_seq_length': CONFIG['max_seq_length']
        }
    
    if len(models_to_analyze) == 0:
        print("\n[ERROR] No models found! Please train models first.")
        print("Run train_models.py to train the models.")
        return
    
    print(f"\n[OK] Loaded {len(models_to_analyze)} model(s) for analysis")
    
    # Step 3: Evaluate and generate qualitative results
    print("\n[3/4] Evaluating models and generating qualitative results...")
    
    all_results = {}
    timing_info = {}
    
    for model_name, model in models_to_analyze:
        print(f"\n{'='*80}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*80}")
        
        # Create evaluator
        evaluator = ModelEvaluator(model=model, model_name=model_name)
        
        # Evaluate model (use subset for faster evaluation)
        print("\nEvaluating model performance...")
        eval_samples = min(len(X1_test), CONFIG['max_eval_samples'])
        if len(X1_test) > eval_samples:
            print(f"  [INFO] Using {eval_samples} samples for evaluation (out of {len(X1_test)}) for faster processing", flush=True)
            eval_indices = np.random.choice(len(X1_test), eval_samples, replace=False)
            X1_eval = X1_test[eval_indices]
            X2_eval = X2_test[eval_indices]
            y_eval = y_test[eval_indices]
        else:
            X1_eval = X1_test
            X2_eval = X2_test
            y_eval = y_test
        
        start_time = time.time()
        test_metrics = evaluator.evaluate(X1_eval, X2_eval, y_eval)
        eval_time_sec = time.time() - start_time
        timing_info[model_name] = {
            'evaluation_time_sec': eval_time_sec,
            'throughput_samples_per_sec': float(len(X1_test)) / eval_time_sec if eval_time_sec > 0 else None
        }
        evaluator.print_metrics(test_metrics)
        
        # Get qualitative results (use subset for faster processing)
        print(f"\nGenerating qualitative results (showing {CONFIG['num_examples']} examples)...")
        # Use larger batch size for faster processing
        batch_size = CONFIG['batch_size_pytorch'] if model_name == 'PyTorch_Attention' else CONFIG['batch_size_keras']
        
        # Use smaller subset for qualitative analysis
        qual_samples = min(len(X1_test), 2000)  # Limit to 2000 samples for qualitative analysis
        if len(X1_test) > qual_samples:
            qual_indices = np.random.choice(len(X1_test), qual_samples, replace=False)
            X1_qual = X1_test[qual_indices]
            X2_qual = X2_test[qual_indices]
            y_qual = y_test[qual_indices]
            pairs_qual = [test_pairs_text[i] for i in qual_indices]
        else:
            X1_qual = X1_test
            X2_qual = X2_test
            y_qual = y_test
            pairs_qual = test_pairs_text
        
        correct_examples, incorrect_examples = evaluator.get_qualitative_results(
            X1_qual, X2_qual, y_qual, pairs_qual, 
            num_examples=CONFIG['num_examples'], batch_size=batch_size
        )
        
        # Display results
        print(f"\n{'-'*80}")
        print(f"Qualitative Results for {model_name}")
        print(f"{'-'*80}")
        
        if len(correct_examples) > 0:
            print(f"\n✓ Correct Predictions (Sample - {len(correct_examples)} examples):")
            display_cols = ['clause1', 'clause2', 'true_label', 'predicted_label', 'probability']
            print(correct_examples[display_cols].to_string(index=False))
        else:
            print("\n[WARNING] No correct predictions found in sample")
        
        if len(incorrect_examples) > 0:
            print(f"\n✗ Incorrect Predictions (Sample - {len(incorrect_examples)} examples):")
            print(incorrect_examples[display_cols].to_string(index=False))
        else:
            print("\n[INFO] No incorrect predictions found in sample (perfect accuracy!)")
        
        # Store results
        all_results[model_name] = {
            'metrics': test_metrics,
            'correct_examples': correct_examples,
            'incorrect_examples': incorrect_examples
        }
    
    # Step 4: Save results
    print("\n[4/4] Saving qualitative results...")
    
    for model_name, results in all_results.items():
        # Save correct predictions
        correct_file = os.path.join(CONFIG['results_dir'], f'{model_name}_correct_predictions.csv')
        results['correct_examples'].to_csv(correct_file, index=False)
        print(f"  [SAVED] {correct_file}")
        
        # Save incorrect predictions
        incorrect_file = os.path.join(CONFIG['results_dir'], f'{model_name}_incorrect_predictions.csv')
        results['incorrect_examples'].to_csv(incorrect_file, index=False)
        print(f"  [SAVED] {incorrect_file}")
        
        # Skip full results generation to save time (can be enabled if needed)
        if not CONFIG['skip_full_results']:
            print(f"\n  Generating full results for {model_name}...")
            # Limit to max 5000 examples to avoid extremely long processing
            max_full_examples = min(len(X1_test), 5000)
            if len(X1_test) > max_full_examples:
                print(f"  [INFO] Limiting full results to {max_full_examples} examples (out of {len(X1_test)})", flush=True)
                indices = np.random.choice(len(X1_test), max_full_examples, replace=False)
                X1_full = X1_test[indices]
                X2_full = X2_test[indices]
                y_full = y_test[indices]
                pairs_full = [test_pairs_text[i] for i in indices]
            else:
                X1_full = X1_test
                X2_full = X2_test
                y_full = y_test
                pairs_full = test_pairs_text
            
            full_evaluator = ModelEvaluator(model=model, model_name=model_name)
            full_correct, full_incorrect = full_evaluator.get_qualitative_results(
                X1_full, X2_full, y_full, pairs_full, num_examples=max_full_examples, batch_size=batch_size
            )
            
            full_correct_file = os.path.join(CONFIG['results_dir'], f'{model_name}_all_correct.csv')
            full_correct.to_csv(full_correct_file, index=False)
            print(f"  [SAVED] {full_correct_file} ({len(full_correct)} correct predictions)")
            
            full_incorrect_file = os.path.join(CONFIG['results_dir'], f'{model_name}_all_incorrect.csv')
            full_incorrect.to_csv(full_incorrect_file, index=False)
            print(f"  [SAVED] {full_incorrect_file} ({len(full_incorrect)} incorrect predictions)")
        else:
            print(f"  [SKIPPED] Full results generation (set CONFIG['skip_full_results']=False to enable)", flush=True)

    # Load and save training graphs if histories exist
    print("\nAttempting to load training histories for graphs...")
    visualizer = TrainingVisualizer(output_dir=CONFIG['results_dir'])
    histories_loaded = False

    # BiLSTM history (JSON or CSV)
    bilstm_hist_json = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_Siamese_history.json')
    bilstm_hist_csv = os.path.join(CONFIG['checkpoint_dir'], 'BiLSTM_training_history.csv')
    bilstm_history = None
    if os.path.exists(bilstm_hist_json):
        try:
            with open(bilstm_hist_json, 'r') as f:
                bilstm_history = json.load(f)
        except Exception:
            bilstm_history = None
    elif os.path.exists(bilstm_hist_csv):
        try:
            df = pd.read_csv(bilstm_hist_csv)
            bilstm_history = {col: df[col].tolist() for col in df.columns if col}
        except Exception:
            bilstm_history = None
    if bilstm_history:
        histories_loaded = True
        visualizer.plot_training_history(bilstm_history, 'BiLSTM_Siamese')

    # PyTorch Attention history (JSON or CSV)
    pt_hist_json = os.path.join(CONFIG['checkpoint_dir'], 'attention_history.json')
    pt_hist_csv = os.path.join(CONFIG['checkpoint_dir'], 'PyTorch_Attention_training_history.csv')
    pt_history = None
    if os.path.exists(pt_hist_json):
        try:
            with open(pt_hist_json, 'r') as f:
                pt_history = json.load(f)
        except Exception:
            pt_history = None
    elif os.path.exists(pt_hist_csv):
        try:
            df = pd.read_csv(pt_hist_csv)
            pt_history = {col: df[col].tolist() for col in df.columns if col}
        except Exception:
            pt_history = None
    if pt_history:
        histories_loaded = True
        # Normalize keys if needed
        normalized_pt_history = {
            'loss': pt_history.get('loss') or pt_history.get('train_loss'),
            'accuracy': pt_history.get('accuracy') or pt_history.get('train_accuracy'),
            'val_loss': pt_history.get('val_loss'),
            'val_accuracy': pt_history.get('val_accuracy'),
        }
        visualizer.plot_training_history(normalized_pt_history, 'PyTorch_Attention')

    if histories_loaded and bilstm_history and pt_history:
        visualizer.plot_loss_accuracy_combined({
            'BiLSTM': bilstm_history,
            'PyTorch_Attention': normalized_pt_history
        })
        # Metrics comparison plot requires metric dicts; reuse test metrics
        metrics_dict = {}
        for model_name, res in all_results.items():
            metrics_dict[model_name] = res['metrics']
        visualizer.plot_metrics_comparison(metrics_dict)
        print("  [OK] Training graphs generated in qualitative_results/")
    else:
        print("  [INFO] Training histories not fully available. Skipping some graphs.")
    
    # Create summary report
    print("\n" + "="*80)
    print("Qualitative Analysis Summary")
    print("="*80)
    
    summary_data = []
    for model_name, results in all_results.items():
        metrics = results['metrics']
        correct_count = len(results['correct_examples'])
        incorrect_count = len(results['incorrect_examples'])
        
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Correct (Sample)': correct_count,
            'Incorrect (Sample)': incorrect_count
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(CONFIG['results_dir'], 'qualitative_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n[SAVED] Summary saved to: {summary_file}")

    # Create a concise markdown report covering rubric items
    report_lines = []
    report_lines.append("# Legal Clause Similarity - Qualitative Analysis Report\n")
    report_lines.append("## Dataset Splits\n")
    report_lines.append(f"- Training pairs: {len(X1_train)}\n- Validation pairs: {len(X1_val)}\n- Test pairs: {len(X1_test)}\n")

    report_lines.append("## Network Details\n")
    for name, details in model_details.items():
        report_lines.append(f"### {name}\n")
        report_lines.append(f"- Architecture: {details.get('architecture','N/A')}\n")
        report_lines.append(f"- Parameters: {details.get('parameters','N/A')}\n")
        report_lines.append(f"- Embedding dim: {details.get('embedding_dim','N/A')}\n")
        report_lines.append(f"- Max seq length: {details.get('max_seq_length','N/A')}\n")

    report_lines.append("## Performance Measures\n")
    report_lines.append(summary_df.to_markdown(index=False))

    report_lines.append("\n## Comparative Analysis (Accuracy & Time)\n")
    comp_rows = []
    for name, res in all_results.items():
        metrics = res['metrics']
        t = timing_info.get(name, {})
        comp_rows.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'PR-AUC': f"{metrics['pr_auc']:.4f}",
            'Eval Time (s)': f"{t.get('evaluation_time_sec', 0):.2f}",
            'Throughput (samples/s)': f"{t.get('throughput_samples_per_sec', 0):.2f}"
        })
    comp_df = pd.DataFrame(comp_rows)
    report_lines.append(comp_df.to_markdown(index=False))

    report_lines.append("\n## Qualitative Results\n")
    for name, res in all_results.items():
        report_lines.append(f"### {name}\n")
        ce = res['correct_examples'].head(5)
        ie = res['incorrect_examples'].head(5)
        if len(ce) > 0:
            report_lines.append("Correct Predictions (sample):\n")
            report_lines.append(ce[['clause1','clause2','true_label','predicted_label','probability']].to_markdown(index=False))
        if len(ie) > 0:
            report_lines.append("\nIncorrect Predictions (sample):\n")
            report_lines.append(ie[['clause1','clause2','true_label','predicted_label','probability']].to_markdown(index=False))
        report_lines.append("\n")

    report_path = os.path.join(CONFIG['results_dir'], 'qualitative_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"[SAVED] Markdown report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("Qualitative Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {CONFIG['results_dir']}/")
    print("\nFiles generated:")
    for model_name in all_results.keys():
        print(f"  - {model_name}_correct_predictions.csv (sample)")
        print(f"  - {model_name}_incorrect_predictions.csv (sample)")
        print(f"  - {model_name}_all_correct.csv (all correct)")
        print(f"  - {model_name}_all_incorrect.csv (all incorrect)")
    print(f"  - qualitative_analysis_summary.csv (summary)")

if __name__ == "__main__":
    main()

