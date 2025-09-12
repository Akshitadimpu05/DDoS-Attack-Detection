#!/usr/bin/env python3
"""
Dataset downloader for Bot-IoT dataset using KaggleHub
"""

import os
import shutil
import pandas as pd
import kagglehub

def download_bot_iot_dataset():
    """Download Bot-IoT dataset using KaggleHub"""
    
    print("Downloading Bot-IoT dataset using KaggleHub...")
    print("This may take a few minutes depending on your internet connection.")
    
    try:
        # Download the dataset using kagglehub
        path = kagglehub.dataset_download("vigneshvenkateswaran/bot-iot")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create local data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Copy files from kagglehub cache to local data directory
        if os.path.exists(path):
            for file in os.listdir(path):
                src_file = os.path.join(path, file)
                dst_file = os.path.join(data_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"  Copied: {file}")
        
        # List downloaded files
        print("\nFiles in data directory:")
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.1f} MB)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return False

def explore_dataset():
    """Explore the downloaded dataset structure"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("❌ Data directory not found. Please download the dataset first.")
        return
    
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\n📁 File: {csv_file}")
        
        try:
            # Read first few rows
            df = pd.read_csv(file_path, nrows=1000)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check for attack/label columns
            label_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['label', 'attack', 'category', 'class'])]
            if label_cols:
                print(f"   Label columns: {label_cols}")
                for col in label_cols:
                    print(f"   {col} unique values: {df[col].unique()[:10]}")
            
        except Exception as e:
            print(f"   ❌ Error reading {csv_file}: {e}")

if __name__ == "__main__":
    print("Bot-IoT Dataset Downloader")
    print("="*30)
    
    success = download_bot_iot_dataset()
    if success:
        explore_dataset()
        print("\n✅ Dataset ready for preprocessing!")
        print("Next step: Run 'python preprocess_data.py' to prepare the data for training.")
    else:
        print("\n❌ Dataset download failed. Please check your Kaggle credentials.")
