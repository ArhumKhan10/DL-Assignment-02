# Legal Clause Similarity Detection - Assignment 2

**Deep Learning Course - NUCES FAST, Islamabad**  
**Instructors:** Mahnoor Tariq & Dr. Qurat Ul Ain

## Overview

This project implements semantic similarity detection for legal clauses using two baseline neural network architectures:

1. **BiLSTM-based Siamese Network**: Uses bidirectional LSTM encoders with shared weights to encode clause pairs and compute similarity.
2. **Simple LSTM-based Siamese Network**: Uses unidirectional LSTM encoders with shared weights. Simpler and faster than BiLSTM, processes sequences in one direction only.

Both models are trained from scratch without using pre-trained transformers or fine-tuned legal models.

## Project Structure

```
Assignment2/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # Model architectures
│   ├── trainer.py          # Training pipeline
│   ├── evaluator.py        # Evaluation metrics
│   └── visualizer.py       # Visualization utilities
├── Legal_Clause_Similarity_Assignment.ipynb  # Main notebook
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Dataset directory (create this)
├── checkpoints/           # Saved models (auto-created)
└── plots/                 # Training plots (auto-created)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset is available on Kaggle:
- **Link**: https://www.kaggle.com/datasets/bahushruth/legalclausedataset

**Easiest way to download (Recommended):**
```bash
python download_dataset.py
```

This script uses `kagglehub` to automatically download and extract the dataset to the `data/` directory.

**Alternative methods:**

1. **Using Kaggle API**:
   - Install: `pip install kaggle`
   - Set up credentials (place `kaggle.json` in `~/.kaggle/` or `C:\Users\<username>\.kaggle\`)
   - Download: `kaggle datasets download -d bahushruth/legalclausedataset`
   - Extract: `unzip legalclausedataset.zip -d data/`

2. **Manual download**: Visit the Kaggle link above and manually extract CSV files to the `data/` directory.

### 3. Dataset Structure

The dataset consists of multiple CSV files, each representing a clause category:
- `acceleration.csv`
- `access-to-information.csv`
- `accounting-terms.csv`
- ... (and more)

Each CSV file should contain:
- A column with clause text
- A column with clause type label

## Usage

### Training Models

**Option 1: Using the training script (Recommended)**
```bash
python train_models.py
```

This script will:
- Load and preprocess the dataset
- Train BiLSTM model (or load from checkpoint if already trained)
- Train Simple LSTM model
- Evaluate both models
- Generate qualitative results
- Save all results and visualizations

**Option 2: Using the notebook**
1. Open `Legal_Clause_Similarity_Assignment.ipynb` in Jupyter Notebook or Google Colab
2. Update the `CONFIG` dictionary in Section 2 if needed:
   - `data_dir`: Path to your dataset directory
   - `max_seq_length`: Maximum sequence length (default: 128)
   - `batch_size`: Training batch size (default: 32)
   - `epochs`: Number of training epochs (default: 50)
3. Run all cells sequentially

### Qualitative Analysis (Standalone)

After models have been trained, you can run qualitative analysis independently:

```bash
python qualitative_analysis.py
```

This script will:
- Load pre-trained models from checkpoints
- Evaluate models on test data
- Generate qualitative results (correct/incorrect predictions)
- Save detailed results to `qualitative_results/` directory:
  - Sample correct/incorrect predictions (10 examples each)
  - All correct/incorrect predictions (complete test set)
  - Summary report with performance metrics

### Configuration Options

```python
CONFIG = {
    'data_dir': 'data',              # Dataset directory
    'max_seq_length': 128,           # Max sequence length
    'embedding_dim': 128,            # Word embedding dimension
    'batch_size': 32,                # Batch size
    'epochs': 50,                    # Training epochs
    'learning_rate': 0.001,          # Learning rate
    'test_size': 0.2,                # Test set proportion
    'val_size': 0.1,                 # Validation set proportion
    'min_word_freq': 2,              # Minimum word frequency for vocabulary
}
```

## Model Architectures

### 1. BiLSTM Siamese Network

- **Encoder**: Bidirectional LSTM with 128 units
- **Embedding**: 128-dimensional word embeddings
- **Similarity Features**: Absolute difference, element-wise product
- **Classifier**: Dense layers with dropout

### 2. Simple LSTM Siamese Network

- **Encoder**: Unidirectional LSTM with 128 units (processes sequences left-to-right only)
- **Embedding**: 128-dimensional word embeddings
- **Similarity Features**: Absolute difference, element-wise product
- **Classifier**: Dense layers with dropout
- **Advantage**: Faster training than BiLSTM, lower memory usage

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Out of predicted similar pairs, how many are truly similar
- **Recall**: Out of all truly similar pairs, how many were identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve

## Output Files

After running the notebook, you'll get:

1. **Models**: Saved in `checkpoints/` directory
   - `BiLSTM_Siamese_best.h5`
   - `Simple_LSTM_Siamese_best.h5`

2. **Training History**: JSON/CSV files in `checkpoints/` directory
   - `BiLSTM_Siamese_history.json`
   - `Simple_LSTM_training_history.csv`
   - `training_results.csv` (comparative results)

3. **Visualizations**: Plots in `plots/` directory
   - Training loss and accuracy curves
   - Metrics comparison charts
   - Combined model comparisons

4. **Qualitative Results**: CSV files in `qualitative_results/` directory (generated by `qualitative_analysis.py`)
   - Sample correct/incorrect predictions (10 examples each)
   - All correct/incorrect predictions (complete test set)
   - Summary report with performance metrics

## Code Features

- **Modular Design**: Separate modules for data loading, models, training, evaluation, and visualization
- **Object-Oriented**: Classes for each component following best practices
- **Well-Documented**: Comprehensive docstrings and comments
- **Reproducible**: Fixed random seeds for consistent results
- **Best Practices**: Follows keras-idiomatic-programmer guidelines

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib
- scikit-learn
- Jupyter Notebook

## Notes

- The models are trained from scratch (no pre-trained embeddings)
- Training time depends on dataset size and hardware
- For large datasets, consider adjusting `num_pairs_per_category` in config
- Early stopping and learning rate reduction are implemented to prevent overfitting

## License

This project is for educational purposes as part of the Deep Learning course assignment.

## Author

**Assignment 2 - CS452**  
NUCES FAST, Islamabad

