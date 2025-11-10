# Legal Clause Similarity Detection - Assignment 2

**Deep Learning Course - NUCES FAST, Islamabad**  
## Overview

This project implements semantic similarity detection for legal clauses using two baseline neural network architectures:

1. **BiLSTM-based Siamese Network (TensorFlow/Keras)**: Uses bidirectional LSTM encoders with shared weights to encode clause pairs and compute similarity.
2. **Attention-based Siamese Network (PyTorch)**: Uses bidirectional LSTM with multi-head attention mechanism for enhanced semantic understanding.

Both models are trained from scratch without using pre-trained transformers or fine-tuned legal models, following assignment requirements.

## Project Structure

```
Assignment2/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading, preprocessing, and pair generation
│   ├── models.py               # TensorFlow/Keras model architectures
│   ├── pytorch_models.py       # PyTorch model architectures
│   ├── trainer.py              # TensorFlow/Keras training pipeline
│   ├── pytorch_trainer.py     # PyTorch training pipeline
│   ├── pytorch_loader.py      # PyTorch model loading utilities
│   ├── evaluator.py           # Evaluation metrics and qualitative analysis
│   └── visualizer.py          # Visualization utilities
├── train_models.py            # Main training script
├── qualitative_analysis.py    # Standalone qualitative analysis script
├── download_dataset.py        # Dataset download utility
├── build_vocabulary.py       # Vocabulary building utility
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Dataset directory (auto-created)
├── checkpoints/              # Saved models and training histories
│   ├── BiLSTM_Siamese_best.h5
│   ├── BiLSTM_Siamese_history.json
│   ├── attention_history.json
│   ├── vocabulary.pkl
│   └── pairs_cache.npz       # Cached pairs for faster reruns
├── qualitative_results/      # Analysis results and visualizations
│   ├── *_correct_predictions.csv
│   ├── *_incorrect_predictions.csv
│   ├── *_training_history.png
│   ├── metrics_comparison.png
│   └── qualitative_analysis_summary.csv
```

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/ArhumKhan10/DL-Assignment-02.git
cd DL-Assignment-02
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- TensorFlow >= 2.10.0
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- kagglehub >= 0.3.0
- Pillow >= 9.0.0

### 4. Download Dataset

**Easiest way (Recommended):**
```bash
python download_dataset.py
```

This script uses `kagglehub` to automatically download and extract the dataset to the `data/` directory.

**Setup Kaggle API (if needed):**
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json` file
4. Place it in:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

**Alternative methods:**
- Manual download from [Kaggle Dataset](https://www.kaggle.com/datasets/bahushruth/legalclausedataset)
- Extract CSV files to the `data/` directory

### 5. Dataset Structure

The dataset consists of 395 CSV files, each representing a clause category:
- `acceleration.csv`
- `access-to-information.csv`
- `accounting-terms.csv`
- ... (and more)

Each CSV file contains:
- Clause text column
- Clause type label column

**Dataset Statistics:**
- Total clauses: 150,881
- Total categories: 395
- Vocabulary size: 58,388 words
- Training pairs: 275,947
- Validation pairs: 39,421
- Test pairs: 78,842

## Usage

### Training Models

**Train both models:**
```bash
python train_models.py
```

This script will:
- Load and preprocess the dataset
- Build vocabulary and generate pairs
- Train BiLSTM model (or load from checkpoint if exists)
- Train PyTorch Attention model
- Evaluate both models on test set
- Save checkpoints and training histories
- Generate performance metrics

**Note:** The script automatically skips training if checkpoints exist. To retrain, delete the checkpoint files.

### Qualitative Analysis

After models have been trained, run qualitative analysis:

```bash
python qualitative_analysis.py
```

This script will:
- Load pre-trained models from checkpoints
- Evaluate models on test data
- Generate qualitative results (correct/incorrect predictions)
- Create training history visualizations
- Save results to `qualitative_results/` directory:
  - Sample correct/incorrect predictions (10 examples each)
  - Training history graphs (PNG)
  - Metrics comparison charts
  - Summary CSV with all performance metrics


### Utility Scripts

**Build Vocabulary Only:**
```bash
python build_vocabulary.py
```

Builds and saves vocabulary without training models. Useful for ensuring vocabulary consistency.

## Model Architectures

### 1. BiLSTM Siamese Network (TensorFlow/Keras)

- **Framework**: TensorFlow/Keras
- **Architecture**: Bidirectional LSTM-based Siamese Network
- **Encoder**: Bidirectional LSTM with 128 units
- **Embedding**: 128-dimensional word embeddings (trainable)
- **Similarity Features**: Absolute difference, element-wise product
- **Classifier**: Dense layers (64 → 32 → 1) with dropout (0.3)
- **Total Parameters**: 7,765,729
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam (learning rate: 0.001)

**Key Features:**
- Captures bidirectional context (past and future tokens)
- Shared encoder weights for both clauses
- Early stopping with target accuracy (0.995)

### 2. Attention-based Siamese Network (PyTorch)

- **Framework**: PyTorch
- **Architecture**: Attention-based Siamese Network with BiLSTM
- **Encoder**: Bidirectional LSTM (128 units) + Multi-head Attention (8 heads)
- **Embedding**: 128-dimensional word embeddings (trainable)
- **Attention**: Multi-head self-attention mechanism
- **Pooling**: Global max pooling over sequence
- **Similarity Features**: Absolute difference, element-wise product
- **Classifier**: Dense layers (64 → 32 → 1) with dropout (0.3)
- **Total Parameters**: 8,029,921
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam (learning rate: 0.001)

**Key Features:**
- Attention mechanism focuses on relevant parts of clauses
- Better interpretability through attention weights
- Enhanced semantic understanding
- Early stopping with plateau detection

## Performance Results

### Quantitative Metrics (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **BiLSTM_Siamese** | 0.9480 | 0.9047 | 0.9988 | 0.9494 | 0.9802 |
| **PyTorch_Attention** | 0.9714 | 0.9500 | 0.9950 | 0.9720 | 0.9900 |

### Key Observations

- **PyTorch Attention** achieves higher accuracy (97.14% vs 94.80%)
- Both models show excellent performance with F1-Scores > 0.94
- Attention mechanism provides better precision (95.00% vs 90.47%)
- BiLSTM has slightly higher recall (99.88% vs 99.50%)
- Both models demonstrate strong ROC-AUC scores (> 0.98)

### Strengths and Weaknesses

**BiLSTM Siamese:**
- Strengths: High recall, faster training, simpler architecture
- Weaknesses: Lower precision, may over-predict similarity

**PyTorch Attention:**
- Strengths: Higher accuracy, better precision, attention interpretability
- Weaknesses: More complex, slower training, more parameters

## Evaluation Metrics

The models are evaluated using comprehensive NLP metrics:

- **Accuracy**: Overall classification correctness (suitable for balanced dataset)
- **Precision**: Out of predicted similar pairs, how many are truly similar (important when false positives are costly)
- **Recall**: Out of all truly similar pairs, how many were identified (critical for finding all similar clauses)
- **F1-Score**: Harmonic mean of precision and recall (balanced metric for NLP tasks)
- **ROC-AUC**: Area under ROC curve (measures ranking ability across thresholds)
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced datasets)

**For production systems**, F1-Score and ROC-AUC are most suitable as they provide balanced performance assessment.

## Configuration

Default configuration in `train_models.py`:

```python
CONFIG = {
    'data_dir': 'data',              # Dataset directory
    'max_seq_length': 128,           # Max sequence length
    'embedding_dim': 128,            # Word embedding dimension
    'batch_size': 32,                # Batch size (Keras)
    'epochs': 5,                     # Training epochs
    'learning_rate': 0.001,          # Learning rate
    'test_size': 0.2,                # Test set proportion
    'val_size': 0.1,                 # Validation set proportion
    'min_word_freq': 2,              # Minimum word frequency for vocabulary
    'checkpoint_dir': 'checkpoints', # Checkpoint directory
}
```

**Note:** PyTorch Attention model uses:
- Batch size: 128 (for faster training)
- Training data subset: 20% (for time constraints)

## Output Files

### Checkpoints Directory

- `BiLSTM_Siamese_best.h5` - Best BiLSTM model weights
- `BiLSTM_Siamese_history.json` - BiLSTM training history
- `attention_history.json` - PyTorch Attention training history
- `vocabulary.pkl` - Saved vocabulary for consistency
- `pairs_cache.npz` / `pairs_text_cache.pkl` - Cached pairs for faster reruns

### Qualitative Results Directory

- `*_correct_predictions.csv` - Sample correct predictions (10 examples)
- `*_incorrect_predictions.csv` - Sample incorrect predictions (10 examples)
- `*_training_history.png` - Training loss and accuracy curves
- `combined_training_comparison.png` - Side-by-side model comparison
- `metrics_comparison.png` - Metrics comparison charts
- `qualitative_analysis_summary.csv` - Summary of all metrics

## Code Features

- **Modular Design**: Separate modules for data loading, models, training, evaluation
- **Object-Oriented**: Classes for each component following best practices
- **Well-Documented**: Comprehensive docstrings and comments
- **Reproducible**: Fixed random seeds for consistent results
- **Best Practices**: Follows keras-idiomatic-programmer and PyTorch best practices
- **Cross-Framework**: Supports both TensorFlow/Keras and PyTorch
- **Efficient**: Caching mechanisms for faster reruns
- **Robust**: Error handling and progress tracking

## Training Features

- **Early Stopping**: Stops training when target accuracy (0.995) is reached
- **Plateau Detection**: Stops if accuracy doesn't improve for 3 epochs
- **Model Checkpointing**: Saves best model during training
- **Learning Rate Reduction**: Automatic LR reduction on plateau
- **Progress Tracking**: Detailed logging with batch-level progress
- **Vocabulary Consistency**: Ensures same vocabulary for training and evaluation

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib
- scikit-learn
- kagglehub (for dataset download)
- Pillow (for PDF generation)

## Notes

- Models are trained from scratch (no pre-trained embeddings)
- Training time depends on dataset size and hardware
- For faster training, the PyTorch model uses 20% of training data
- Early stopping prevents overfitting
- Vocabulary is saved to ensure consistency between training and evaluation
- Cache files speed up subsequent runs

## License

This project is for educational purposes as part of the Deep Learning course assignment.