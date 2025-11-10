"""
Script to download the legal clause dataset from Kaggle using kagglehub.

This script downloads the dataset from:
https://www.kaggle.com/datasets/bahushruth/legalclausedataset

Note: You may need to authenticate with Kaggle first.
If authentication is required, you'll be prompted to do so.
"""

import os
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub is not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

def download_dataset():
    """Download the legal clause dataset from Kaggle."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Downloading Legal Clause Dataset from Kaggle")
    print("="*60)
    print("Dataset: bahushruth/legalclausedataset")
    print("\nAttempting to download...")
    
    try:
        # Download latest version using kagglehub
        path = kagglehub.dataset_download("bahushruth/legalclausedataset")
        
        print(f"\nDataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # Copy files to data directory
        source_path = Path(path)
        if source_path.exists():
            print(f"\nCopying files to {data_dir}...")
            
            # Copy all CSV files to data directory
            csv_files = list(source_path.glob("*.csv"))
            if csv_files:
                for csv_file in csv_files:
                    dest_file = data_dir / csv_file.name
                    shutil.copy2(csv_file, dest_file)
                    print(f"  Copied: {csv_file.name}")
                
                print(f"\n[SUCCESS] Successfully copied {len(csv_files)} CSV files to {data_dir}/")
                print(f"[SUCCESS] Dataset is ready to use!")
                return True
            else:
                # If no CSV files found, check subdirectories
                for subdir in source_path.iterdir():
                    if subdir.is_dir():
                        csv_files = list(subdir.glob("*.csv"))
                        for csv_file in csv_files:
                            dest_file = data_dir / csv_file.name
                            shutil.copy2(csv_file, dest_file)
                            print(f"  Copied: {csv_file.name}")
                
                if list(data_dir.glob("*.csv")):
                    print(f"\n[SUCCESS] Successfully copied CSV files to {data_dir}/")
                    print(f"[SUCCESS] Dataset is ready to use!")
                    return True
                else:
                    print(f"\n[WARNING] No CSV files found in {path}")
                    print(f"Please check the dataset structure manually.")
                    return False
        else:
            print(f"\n[WARNING] Downloaded path does not exist: {path}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error downloading dataset: {e}")
        print("\n" + "="*60)
        print("TROUBLESHOOTING:")
        print("="*60)
        print("1. Make sure you have a Kaggle account")
        print("2. You may need to authenticate:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Scroll to 'API' section")
        print("   - Click 'Create New API Token'")
        print("   - Place kaggle.json in: C:/Users/<your_username>/.kaggle/")
        print("\nOr manually download from:")
        print("https://www.kaggle.com/datasets/bahushruth/legalclausedataset")
        print("and extract CSV files to the 'data' directory")
        print("="*60)
        return False

if __name__ == "__main__":
    download_dataset()

