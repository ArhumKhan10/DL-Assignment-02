"""
Data Loading and Preprocessing Module

This module handles:
- Loading legal clause datasets from CSV files
- Creating positive and negative pairs for similarity learning
- Text preprocessing and tokenization
- Dataset splitting and batching
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import re
from sklearn.model_selection import train_test_split


class LegalClauseDataLoader:
    """
    Handles loading and preprocessing of legal clause datasets.
    
    Attributes:
        data_dir (str): Directory containing CSV files
        vocab (dict): Vocabulary mapping words to indices
        reverse_vocab (dict): Reverse mapping from indices to words
        max_seq_length (int): Maximum sequence length for padding
        word_to_idx (dict): Word to index mapping
        idx_to_word (dict): Index to word mapping
    """
    
    def __init__(self, data_dir: str, max_seq_length: int = 128):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing CSV files
            max_seq_length: Maximum sequence length for padding
        """
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.clauses_by_category = {}
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:()\[\]{}]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def load_dataset(self) -> Dict[str, List[str]]:
        """
        Load all CSV files from the data directory.
        
        Returns:
            Dictionary mapping category names to lists of clause texts
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        clauses_by_category = {}
        
        # Get all CSV files first
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        total_files = len(csv_files)
        print(f"Found {total_files} CSV files to load...")
        
        # Load each CSV file
        for file_idx, filename in enumerate(csv_files):
            category = filename.replace('.csv', '')
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                df = pd.read_csv(filepath)
                # Assume columns are 'clause' and 'clause_type' or similar
                # Try common column names
                text_col = None
                for col in df.columns:
                    if 'clause' in col.lower() and 'type' not in col.lower():
                        text_col = col
                        break
                
                if text_col is None:
                    # Use first column as text
                    text_col = df.columns[0]
                
                clauses = df[text_col].astype(str).tolist()
                # Clean clauses
                clauses = [self.clean_text(clause) for clause in clauses if pd.notna(clause)]
                clauses_by_category[category] = clauses
                
                if (file_idx + 1) % 50 == 0 or (file_idx + 1) == total_files:
                    print(f"Loaded {file_idx + 1}/{total_files} files... ({len(clauses)} clauses from {category})", flush=True)
                elif (file_idx + 1) % 10 == 0:
                    print(f"Loaded {file_idx + 1}/{total_files} files...", flush=True)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        print(f"\n[OK] Finished loading all {total_files} files", flush=True)
        total_clauses = sum(len(clauses) for clauses in clauses_by_category.values())
        print(f"[OK] Total clauses loaded: {total_clauses}", flush=True)
        self.clauses_by_category = clauses_by_category
        return clauses_by_category
    
    def build_vocabulary(self, min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from all clauses.
        
        Args:
            min_freq: Minimum frequency for a word to be included
            
        Returns:
            Vocabulary dictionary mapping words to indices
        """
        # Collect all words
        print("    - Processing clauses to collect words...", flush=True)
        word_counts = Counter()
        total_clauses = sum(len(clauses) for clauses in self.clauses_by_category.values())
        print(f"    - Total clauses to process: {total_clauses}", flush=True)
        processed = 0
        for cat_idx, (category, clauses) in enumerate(self.clauses_by_category.items()):
            for clause in clauses:
                words = clause.split()
                word_counts.update(words)
                processed += 1
            if (cat_idx + 1) % 50 == 0:
                print(f"    - Processed {cat_idx + 1}/{len(self.clauses_by_category)} categories, "
                      f"{processed}/{total_clauses} clauses...", flush=True)
        print(f"    - Processed all {processed} clauses", flush=True)
        print(f"    - Collected {len(word_counts)} unique words", flush=True)
        
        # Build vocabulary with special tokens
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # Add words that meet minimum frequency
        print(f"    - Building vocabulary (min_freq={min_freq})...", flush=True)
        idx = len(self.word_to_idx)
        words_added = 0
        total_words = len(word_counts)
        processed_words = 0
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word_to_idx[word] = idx
                idx += 1
                words_added += 1
            processed_words += 1
            if processed_words % 5000 == 0:
                print(f"    - Processed {processed_words}/{total_words} words, "
                      f"added {words_added} to vocabulary...", flush=True)
        
        # Create reverse mapping
        print(f"    - Creating reverse mapping...", flush=True)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"    - Added {words_added} words to vocabulary", flush=True)
        print(f"Vocabulary size: {self.vocab_size}", flush=True)
        return self.word_to_idx
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of word indices
        """
        words = text.split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        return sequence
    
    def pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            sequence: List of word indices
            
        Returns:
            Padded/truncated sequence as numpy array
        """
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            padding = [self.word_to_idx['<PAD>']] * (self.max_seq_length - len(sequence))
            sequence = sequence + padding
        
        return np.array(sequence, dtype=np.int32)
    
    def create_pairs(self, num_positive: Optional[int] = None, 
                     num_negative: Optional[int] = None,
                     balance: bool = True) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Create positive and negative pairs for training.
        
        Args:
            num_positive: Number of positive pairs to generate (None = all possible)
            num_negative: Number of negative pairs to generate (None = match positive)
            balance: Whether to balance positive and negative pairs
            
        Returns:
            Tuple of (pairs, labels) where pairs is list of (clause1, clause2) tuples
            and labels is list of 1 (similar) or 0 (not similar)
        """
        pairs = []
        labels = []
        
        categories = list(self.clauses_by_category.keys())
        print(f"    - Processing {len(categories)} categories...", flush=True)
        
        # Generate positive pairs (same category)
        # Limit pairs per category to avoid memory issues
        MAX_PAIRS_PER_CATEGORY = 500  # 500 pairs per category for faster processing
        print(f"    - Max pairs per category: {MAX_PAIRS_PER_CATEGORY}", flush=True)
        
        positive_pairs = []
        total_categories = len(categories)
        print(f"    - Starting pair generation for {total_categories} categories...", flush=True)
        
        for cat_idx, category in enumerate(categories):
            clauses = self.clauses_by_category[category]
            category_pairs = []
            
            # Create pairs within same category (limited)
            max_pairs_for_this_cat = min(MAX_PAIRS_PER_CATEGORY, len(clauses) * (len(clauses) - 1) // 2)
            if max_pairs_for_this_cat > 0:
                # Use random sampling for large categories
                if len(clauses) > 50:  # For large categories, sample randomly
                    indices = np.random.choice(len(clauses), size=min(100, len(clauses)), replace=False)
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            category_pairs.append((clauses[indices[i]], clauses[indices[j]]))
                            if len(category_pairs) >= MAX_PAIRS_PER_CATEGORY:
                                break
                        if len(category_pairs) >= MAX_PAIRS_PER_CATEGORY:
                            break
                else:
                    # For small categories, generate all pairs
                    for i in range(len(clauses)):
                        for j in range(i + 1, len(clauses)):
                            category_pairs.append((clauses[i], clauses[j]))
                            if len(category_pairs) >= MAX_PAIRS_PER_CATEGORY:
                                break
                        if len(category_pairs) >= MAX_PAIRS_PER_CATEGORY:
                            break
            
            positive_pairs.extend(category_pairs)
            
            if (cat_idx + 1) % 25 == 0 or (cat_idx + 1) == total_categories:
                print(f"    - Processed {cat_idx + 1}/{total_categories} categories, "
                      f"generated {len(positive_pairs)} positive pairs so far...", flush=True)
        
        print(f"    - Total positive pairs generated: {len(positive_pairs)}", flush=True)
        
        if num_positive and num_positive < len(positive_pairs):
            print(f"    - Limiting to {num_positive} positive pairs...", flush=True)
            np.random.shuffle(positive_pairs)
            positive_pairs = positive_pairs[:num_positive]
        
        pairs.extend(positive_pairs)
        labels.extend([1] * len(positive_pairs))
        print(f"    - Added {len(positive_pairs)} positive pairs", flush=True)
        
        # Generate negative pairs (different categories)
        # Limit negative pairs to avoid excessive generation time
        MAX_NEGATIVE_PAIRS = 200000  # Cap at 200k negative pairs
        
        num_neg = len(positive_pairs) if balance and num_negative is None else num_negative
        if num_neg is None:
            num_neg = len(positive_pairs)
        
        # Cap the number of negative pairs
        if num_neg > MAX_NEGATIVE_PAIRS:
            print(f"    - Limiting negative pairs from {num_neg} to {MAX_NEGATIVE_PAIRS} for faster processing...", flush=True)
            num_neg = MAX_NEGATIVE_PAIRS
        
        print(f"    - Generating {num_neg} negative pairs...", flush=True)
        negative_pairs = []
        
        # Pre-generate category pairs for faster access
        category_pairs_list = []
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i < j:  # Avoid duplicates
                    category_pairs_list.append((cat1, cat2))
        
        # Generate negative pairs more efficiently
        for neg_idx in range(num_neg):
            # Randomly select two different categories
            if len(category_pairs_list) > 0:
                cat1, cat2 = category_pairs_list[np.random.randint(len(category_pairs_list))]
            else:
                cat1, cat2 = np.random.choice(categories, size=2, replace=False)
            
            clause1 = np.random.choice(self.clauses_by_category[cat1])
            clause2 = np.random.choice(self.clauses_by_category[cat2])
            negative_pairs.append((clause1, clause2))
            
            if (neg_idx + 1) % 10000 == 0:
                print(f"    - Generated {neg_idx + 1}/{num_neg} negative pairs...", flush=True)
        
        pairs.extend(negative_pairs)
        labels.extend([0] * len(negative_pairs))
        print(f"    - Added {len(negative_pairs)} negative pairs", flush=True)
        
        # Shuffle pairs and labels together
        print(f"    - Shuffling {len(pairs)} total pairs...", flush=True)
        indices = np.arange(len(pairs))
        np.random.shuffle(indices)
        pairs = [pairs[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"    - Shuffling complete", flush=True)
        
        print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs", flush=True)
        return pairs, labels
    
    def prepare_data(self, pairs: List[Tuple[str, str]], labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert pairs to model-ready format.
        
        Args:
            pairs: List of (clause1, clause2) tuples
            labels: List of similarity labels (1 or 0)
            
        Returns:
            Tuple of (seq1_array, seq2_array, labels_array)
        """
        seq1_list = []
        seq2_list = []
        
        total_pairs = len(pairs)
        for pair_idx, (clause1, clause2) in enumerate(pairs):
            seq1 = self.text_to_sequence(clause1)
            seq2 = self.text_to_sequence(clause2)
            seq1_padded = self.pad_sequence(seq1)
            seq2_padded = self.pad_sequence(seq2)
            seq1_list.append(seq1_padded)
            seq2_list.append(seq2_padded)
            
            if (pair_idx + 1) % 5000 == 0 or (pair_idx + 1) == total_pairs:
                print(f"    - Processed {pair_idx + 1}/{total_pairs} pairs...", flush=True)
        
        print(f"    - Converting to numpy arrays...", flush=True)
        X1 = np.array(seq1_list)
        X2 = np.array(seq2_list)
        y = np.array(labels, dtype=np.int32)
        print(f"    - Conversion complete", flush=True)
        return X1, X2, y
    
    def split_data(self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray,
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X1: First clause sequences
            X2: Second clause sequences
            y: Labels
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test)
        """
        # First split: train+val vs test
        X1_temp, X1_test, X2_temp, X2_test, y_temp, y_test = train_test_split(
            X1, X2, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1_temp, X2_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X1_train)}, Val: {len(X1_val)}, Test: {len(X1_test)}")
        return X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test

