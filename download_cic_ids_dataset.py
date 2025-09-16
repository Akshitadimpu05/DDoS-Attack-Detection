#!/usr/bin/env python3
"""
Dataset downloader for CIC-IDS-2017 dataset - A comprehensive network intrusion detection dataset
This dataset contains over 2.8 million network flow records with various attack types including DDoS
"""

import os
import shutil
import pandas as pd
import kagglehub
import numpy as np
from pathlib import Path

def download_cic_ids_dataset():
    """Download CIC-IDS-2017 dataset using KaggleHub"""
    
    print("Downloading CIC-IDS-2017 dataset using KaggleHub...")
    print("This is a large dataset (~2.8M records) and may take several minutes to download.")
    
    try:
        # Download the CIC-IDS-2017 dataset
        path = kagglehub.dataset_download("cicdataset/cicids2017")
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
        total_size = 0
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        print(f"\nTotal dataset size: {total_size:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please check your internet connection and Kaggle credentials.")
        return False

def explore_cic_ids_dataset():
    """Explore the downloaded CIC-IDS-2017 dataset structure"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("❌ Data directory not found. Please download the dataset first.")
        return
    
    print("\n" + "="*60)
    print("CIC-IDS-2017 DATASET EXPLORATION")
    print("="*60)
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    total_records = 0
    attack_types = set()
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\n📁 File: {csv_file}")
        
        try:
            # Read the entire file to get accurate statistics
            print("  Loading file... (this may take a moment)")
            df = pd.read_csv(file_path)
            
            print(f"   Shape: {df.shape}")
            total_records += df.shape[0]
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            print(f"   Columns: {len(df.columns)}")
            
            # Check for label columns
            label_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['label', 'attack', 'category', 'class'])]
            
            if label_cols:
                print(f"   Label columns: {label_cols}")
                for col in label_cols:
                    unique_vals = df[col].unique()
                    print(f"   {col} unique values ({len(unique_vals)}): {list(unique_vals)}")
                    attack_types.update(unique_vals)
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"   Missing values: {missing_count}")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   Memory usage: {memory_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Error reading {csv_file}: {e}")
    
    print(f"\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {total_records:,}")
    print(f"Total files: {len(csv_files)}")
    print(f"Attack types found: {len(attack_types)}")
    print(f"Attack categories: {sorted(list(attack_types))}")
    
    # Estimate if this will help with overfitting
    if total_records > 100000:
        print(f"\n✅ This dataset has {total_records:,} records - excellent for preventing overfitting!")
        print("   This is significantly larger than BoT-IoT and should provide better training.")
    else:
        print(f"\n⚠️  Dataset has {total_records:,} records - may still be small for complex models.")

def prepare_dataset_info():
    """Prepare dataset information for preprocessing"""
    
    info = {
        'dataset_name': 'CIC-IDS-2017',
        'description': 'Comprehensive network intrusion detection dataset with DDoS attacks',
        'features': [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ],
        'attack_types': [
            'BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration',
            'Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection',
            'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
            'DoS Hulk', 'DoS GoldenEye', 'Heartbleed'
        ],
        'binary_classification': {
            'normal': ['BENIGN'],
            'attack': ['DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack – Brute Force',
                      'Web Attack – XSS', 'Web Attack – Sql Injection', 'FTP-Patator',
                      'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk',
                      'DoS GoldenEye', 'Heartbleed']
        }
    }
    
    # Save dataset info
    import json
    with open('data/dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✅ Dataset information saved to data/dataset_info.json")
    return info

if __name__ == "__main__":
    print("CIC-IDS-2017 Dataset Downloader")
    print("="*40)
    print("This dataset contains over 2.8 million network flow records")
    print("Perfect for training robust DDoS detection models!")
    print()
    
    success = download_cic_ids_dataset()
    if success:
        explore_cic_ids_dataset()
        prepare_dataset_info()
        print("\n✅ CIC-IDS-2017 dataset ready for preprocessing!")
        print("Next step: Run 'python preprocess_data.py' to prepare the data for training.")
    else:
        print("\n❌ Dataset download failed. Please check your Kaggle credentials.")
        print("Make sure you have kaggle API credentials configured:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API token")
        print("3. Place kaggle.json in ~/.kaggle/")
