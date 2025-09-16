#!/usr/bin/env python3
"""
Dataset downloader for NSL-KDD dataset - A comprehensive network intrusion detection dataset
This dataset is publicly available and doesn't require Kaggle authentication
"""

import os
import shutil
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from pathlib import Path

def download_nsl_kdd_dataset():
    """Download NSL-KDD dataset from public repository"""
    
    print("Downloading NSL-KDD dataset from public repository...")
    print("This dataset contains comprehensive network intrusion data including DDoS attacks.")
    
    try:
        # Create data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # NSL-KDD dataset URLs (publicly available)
        urls = {
            'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'KDDTest+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt',
            'KDDTrain+_20Percent.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt'
        }
        
        # Download files
        for filename, url in urls.items():
            file_path = os.path.join(data_dir, filename)
            print(f"Downloading {filename}...")
            
            try:
                urllib.request.urlretrieve(url, file_path)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✅ Downloaded: {filename} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                # Try alternative approach
                try:
                    import requests
                    response = requests.get(url)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  ✅ Downloaded (alternative): {filename} ({size_mb:.1f} MB)")
                except Exception as e2:
                    print(f"  ❌ Alternative download failed: {e2}")
        
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
        return False

def create_nsl_kdd_csv():
    """Convert NSL-KDD text files to CSV format with proper headers"""
    
    print("Converting NSL-KDD files to CSV format...")
    
    # Define column names for NSL-KDD dataset
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]
    
    data_dir = "data"
    
    # Process training file
    train_file = os.path.join(data_dir, 'KDDTrain+.txt')
    if os.path.exists(train_file):
        print("Processing training data...")
        df_train = pd.read_csv(train_file, header=None, names=column_names)
        df_train.to_csv(os.path.join(data_dir, 'nsl_kdd_train.csv'), index=False)
        print(f"  ✅ Training data: {len(df_train):,} records")
    
    # Process test file
    test_file = os.path.join(data_dir, 'KDDTest+.txt')
    if os.path.exists(test_file):
        print("Processing test data...")
        df_test = pd.read_csv(test_file, header=None, names=column_names)
        df_test.to_csv(os.path.join(data_dir, 'nsl_kdd_test.csv'), index=False)
        print(f"  ✅ Test data: {len(df_test):,} records")
    
    # Process 20% training file (smaller subset for quick testing)
    small_file = os.path.join(data_dir, 'KDDTrain+_20Percent.txt')
    if os.path.exists(small_file):
        print("Processing 20% training data...")
        df_small = pd.read_csv(small_file, header=None, names=column_names)
        df_small.to_csv(os.path.join(data_dir, 'nsl_kdd_train_20percent.csv'), index=False)
        print(f"  ✅ 20% Training data: {len(df_small):,} records")
    
    print("✅ CSV conversion completed!")

def explore_nsl_kdd_dataset():
    """Explore the downloaded NSL-KDD dataset structure"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("❌ Data directory not found. Please download the dataset first.")
        return
    
    print("\n" + "="*60)
    print("NSL-KDD DATASET EXPLORATION")
    print("="*60)
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    total_records = 0
    attack_types = set()
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\n📁 File: {csv_file}")
        
        try:
            # Read sample of the file
            df = pd.read_csv(file_path, nrows=1000)
            
            print(f"   Shape (sample): {df.shape}")
            print(f"   Columns: {len(df.columns)}")
            
            # Get full file size
            df_full = pd.read_csv(file_path)
            total_records += len(df_full)
            print(f"   Total records: {len(df_full):,}")
            
            # Check attack types
            if 'attack_type' in df.columns:
                unique_attacks = df_full['attack_type'].unique()
                print(f"   Attack types ({len(unique_attacks)}): {list(unique_attacks)[:10]}...")
                attack_types.update(unique_attacks)
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"   Missing values: {missing_count}")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   Memory usage (sample): {memory_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Error reading {csv_file}: {e}")
    
    print(f"\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {total_records:,}")
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Attack types found: {len(attack_types)}")
    
    # Show attack categories
    if attack_types:
        normal_attacks = [a for a in attack_types if 'normal' in a.lower()]
        dos_attacks = [a for a in attack_types if any(x in a.lower() for x in ['dos', 'ddos', 'smurf', 'neptune', 'back', 'land', 'pod', 'teardrop'])]
        probe_attacks = [a for a in attack_types if any(x in a.lower() for x in ['probe', 'scan', 'ipsweep', 'nmap', 'portsweep', 'satan'])]
        
        print(f"\nAttack Categories:")
        print(f"  Normal: {normal_attacks}")
        print(f"  DoS/DDoS: {dos_attacks}")
        print(f"  Probe: {probe_attacks}")
        print(f"  Others: {list(attack_types - set(normal_attacks) - set(dos_attacks) - set(probe_attacks))}")
    
    # Estimate if this will help with overfitting
    if total_records > 50000:
        print(f"\n✅ This dataset has {total_records:,} records - excellent for preventing overfitting!")
        print("   This is much larger than BoT-IoT and should provide better training.")
    else:
        print(f"\n⚠️  Dataset has {total_records:,} records - may still be small for complex models.")

def prepare_nsl_kdd_info():
    """Prepare dataset information for preprocessing"""
    
    info = {
        'dataset_name': 'NSL-KDD',
        'description': 'Network Security Laboratory - Knowledge Discovery and Data Mining dataset with DDoS attacks',
        'attack_categories': {
            'normal': ['normal'],
            'dos': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
            'probe': ['ipsweep', 'nmap', 'portsweep', 'satan'],
            'r2l': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
            'u2r': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
        },
        'binary_classification': {
            'normal': ['normal'],
            'attack': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'ipsweep', 'nmap', 
                      'portsweep', 'satan', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 
                      'phf', 'spy', 'warezclient', 'warezmaster', 'buffer_overflow', 'loadmodule', 
                      'perl', 'rootkit']
        }
    }
    
    # Save dataset info
    import json
    os.makedirs('data', exist_ok=True)
    with open('data/dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✅ Dataset information saved to data/dataset_info.json")
    return info

if __name__ == "__main__":
    print("NSL-KDD Dataset Downloader")
    print("="*40)
    print("This dataset contains comprehensive network intrusion data")
    print("Perfect for training robust DDoS detection models!")
    print("No Kaggle authentication required!")
    print()
    
    success = download_nsl_kdd_dataset()
    if success:
        create_nsl_kdd_csv()
        explore_nsl_kdd_dataset()
        prepare_nsl_kdd_info()
        print("\n✅ NSL-KDD dataset ready for preprocessing!")
        print("Next step: Run 'python preprocess_nsl_kdd_data.py' to prepare the data for training.")
    else:
        print("\n❌ Dataset download failed. Please check your internet connection.")
