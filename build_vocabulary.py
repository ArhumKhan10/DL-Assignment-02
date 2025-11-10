"""
Build and Save Vocabulary Script

This script builds the vocabulary from the dataset and saves it to checkpoints/vocabulary.pkl.
This ensures consistency between training and evaluation without needing to retrain models.
"""

import os
import sys
import pickle

# Add src to path
sys.path.append('src')

from data_loader import LegalClauseDataLoader

# Configuration
CONFIG = {
    'data_dir': 'data',
    'checkpoint_dir': 'checkpoints',
    'max_seq_length': 128,
    'min_word_freq': 2
}

def main():
    print("="*80)
    print("Building and Saving Vocabulary")
    print("="*80)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Step 1: Load dataset
    print("\n[1/3] Loading dataset...")
    data_loader = LegalClauseDataLoader(
        data_dir=CONFIG['data_dir'],
        max_seq_length=CONFIG['max_seq_length']
    )
    
    print("  - Loading CSV files...", flush=True)
    clauses_by_category = data_loader.load_dataset()
    print(f"  [OK] Loaded {len(clauses_by_category)} categories", flush=True)
    total_clauses = sum(len(clauses) for clauses in clauses_by_category.values())
    print(f"  [OK] Total clauses: {total_clauses}", flush=True)
    
    # Step 2: Build vocabulary
    print("\n[2/3] Building vocabulary...")
    print(f"  - Minimum word frequency: {CONFIG['min_word_freq']}", flush=True)
    vocab = data_loader.build_vocabulary(min_freq=CONFIG['min_word_freq'])
    vocab_size = data_loader.vocab_size
    print(f"  [OK] Vocabulary size: {vocab_size}", flush=True)
    
    # Step 3: Save vocabulary
    print("\n[3/3] Saving vocabulary...")
    vocab_file = os.path.join(CONFIG['checkpoint_dir'], 'vocabulary.pkl')
    
    vocab_data = {
        'word_to_idx': data_loader.word_to_idx,
        'idx_to_word': data_loader.idx_to_word,
        'vocab_size': data_loader.vocab_size
    }
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print(f"  [OK] Vocabulary saved to {vocab_file}", flush=True)
    print(f"\n  Vocabulary statistics:", flush=True)
    print(f"    - Total words: {vocab_size}", flush=True)
    print(f"    - Special tokens: 4 (<PAD>, <UNK>, <START>, <END>)", flush=True)
    print(f"    - Regular words: {vocab_size - 4}", flush=True)
    
    print("\n" + "="*80)
    print("[DONE] Vocabulary built and saved successfully!")
    print("="*80)
    print("\nYou can now run qualitative_analysis.py and it will use this vocabulary.")
    print("="*80)

if __name__ == '__main__':
    main()

