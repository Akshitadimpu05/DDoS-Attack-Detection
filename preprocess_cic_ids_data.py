#!/usr/bin/env python3
"""
Enhanced preprocessing script for CIC-IDS-2017 dataset
Handles large dataset with proper memory management and prevents overfitting
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import json
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

class CICIDSPreprocessor:
    """Enhanced preprocessor for CIC-IDS-2017 dataset"""
    
    def __init__(self, data_dir="data", processed_dir="processed_data"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Create processed data directory
        os.makedirs(processed_dir, exist_ok=True)
        
        # Dataset configuration
        self.target_column = 'Label'
        self.sequence_length = 100  # Increased for better temporal patterns
        self.step_size = 50  # Reduced overlap to prevent overfitting
        
    def load_dataset(self):
        """Load CIC-IDS-2017 dataset with memory-efficient processing"""
        
        print("Loading CIC-IDS-2017 dataset...")
        
        # Find CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data directory. Please download the dataset first.")
        
        all_data = []
        total_records = 0
        
        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            print(f"Processing {csv_file}...")
            
            try:
                # Load in chunks to handle large files
                chunk_list = []
                chunk_size = 50000
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Clean column names
                    chunk.columns = chunk.columns.str.strip()
                    chunk_list.append(chunk)
                
                df = pd.concat(chunk_list, ignore_index=True)
                all_data.append(df)
                total_records += len(df)
                
                print(f"  Loaded {len(df):,} records from {csv_file}")
                
            except Exception as e:
                print(f"  ❌ Error loading {csv_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded from CSV files.")
        
        # Combine all data
        print("Combining all data files...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"✅ Total dataset loaded: {len(combined_data):,} records")
        return combined_data
    
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        
        print("Cleaning dataset...")
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != self.target_column:
                df[col] = df[col].fillna(df[col].median())
        
        # Clean target column
        if self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].str.strip()
        else:
            # Try alternative label column names
            possible_labels = ['Label', ' Label', 'label', 'attack_type', 'Label ']
            for label_col in possible_labels:
                if label_col in df.columns:
                    self.target_column = label_col
                    df[self.target_column] = df[self.target_column].str.strip()
                    break
            else:
                raise ValueError("No label column found in dataset")
        
        print(f"Dataset shape after cleaning: {df.shape}")
        return df
    
    def create_binary_labels(self, df):
        """Convert multi-class labels to binary (Normal vs Attack)"""
        
        print("Creating binary labels...")
        
        # Get unique labels
        unique_labels = df[self.target_column].unique()
        print(f"Unique labels found: {unique_labels}")
        
        # Create binary labels (0: Normal, 1: Attack)
        normal_labels = ['BENIGN', 'Benign', 'benign', 'NORMAL', 'Normal', 'normal']
        
        df['binary_label'] = 0  # Default to normal
        for label in unique_labels:
            if label not in normal_labels:
                df.loc[df[self.target_column] == label, 'binary_label'] = 1
        
        # Count distribution
        label_counts = df['binary_label'].value_counts()
        print(f"Label distribution:")
        print(f"  Normal (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Attack (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def balance_dataset(self, df, max_samples_per_class=200000):
        """Balance the dataset to prevent class imbalance issues"""
        
        print("Balancing dataset...")
        
        # Separate classes
        normal_data = df[df['binary_label'] == 0]
        attack_data = df[df['binary_label'] == 1]
        
        print(f"Before balancing:")
        print(f"  Normal samples: {len(normal_data):,}")
        print(f"  Attack samples: {len(attack_data):,}")
        
        # Limit samples per class to prevent overfitting and memory issues
        if len(normal_data) > max_samples_per_class:
            normal_data = normal_data.sample(n=max_samples_per_class, random_state=42)
        
        if len(attack_data) > max_samples_per_class:
            attack_data = attack_data.sample(n=max_samples_per_class, random_state=42)
        
        # Balance classes by undersampling majority class
        min_samples = min(len(normal_data), len(attack_data))
        
        if len(normal_data) > min_samples:
            normal_data = normal_data.sample(n=min_samples, random_state=42)
        
        if len(attack_data) > min_samples:
            attack_data = attack_data.sample(n=min_samples, random_state=42)
        
        # Combine balanced data
        balanced_df = pd.concat([normal_data, attack_data], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After balancing:")
        print(f"  Total samples: {len(balanced_df):,}")
        print(f"  Normal samples: {len(balanced_df[balanced_df['binary_label'] == 0]):,}")
        print(f"  Attack samples: {len(balanced_df[balanced_df['binary_label'] == 1]):,}")
        
        return balanced_df
    
    def prepare_features(self, df):
        """Prepare features for CNN-TCN model"""
        
        print("Preparing features...")
        
        # Select numeric features only
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        feature_columns = [col for col in feature_columns if col not in [self.target_column, 'binary_label']]
        
        print(f"Selected {len(feature_columns)} features")
        
        # Extract features and labels
        X = df[feature_columns].values
        y = df['binary_label'].values
        
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y, feature_columns
    
    def create_sequences(self, X, y):
        """Create sequences for CNN-TCN model with reduced overlap"""
        
        print(f"Creating sequences (length={self.sequence_length}, step={self.step_size})...")
        
        sequences = []
        labels = []
        
        # Create sequences with sliding window
        for i in range(0, len(X) - self.sequence_length + 1, self.step_size):
            sequence = X[i:i + self.sequence_length]
            label = y[i + self.sequence_length - 1]  # Use last label in sequence
            
            sequences.append(sequence)
            labels.append(label)
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        print(f"Created {len(X_seq):,} sequences")
        print(f"Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def split_and_scale_data(self, X, y):
        """Split data and apply scaling"""
        
        print("Splitting and scaling data...")
        
        # Split data: 70% train, 15% validation, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"Data split:")
        print(f"  Training: {X_train.shape[0]:,} samples")
        print(f"  Validation: {X_val.shape[0]:,} samples")
        print(f"  Test: {X_test.shape[0]:,} samples")
        
        # Reshape for scaling (flatten sequences)
        original_train_shape = X_train.shape
        original_val_shape = X_val.shape
        original_test_shape = X_test.shape
        
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data only
        print("Fitting scaler...")
        self.scaler.fit(X_train_flat)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(original_train_shape)
        X_val_scaled = self.scaler.transform(X_val_flat).reshape(original_val_shape)
        X_test_scaled = self.scaler.transform(X_test_flat).reshape(original_test_shape)
        
        print("✅ Data scaling completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
        """Save processed data to disk"""
        
        print("Saving processed data...")
        
        # Save arrays
        np.save(os.path.join(self.processed_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.processed_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(self.processed_dir, 'y_test.npy'), y_test)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.processed_dir, 'scaler.pkl'))
        
        # Save feature names and metadata
        metadata = {
            'feature_names': feature_names,
            'sequence_length': self.sequence_length,
            'step_size': self.step_size,
            'n_features': len(feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'input_shape': list(X_train.shape[1:])
        }
        
        with open(os.path.join(self.processed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Processed data saved:")
        print(f"  Training data: {X_train.shape}")
        print(f"  Validation data: {X_val.shape}")
        print(f"  Test data: {X_test.shape}")
        print(f"  Features: {len(feature_names)}")
        
    def process_dataset(self):
        """Main processing pipeline"""
        
        print("="*60)
        print("CIC-IDS-2017 DATASET PREPROCESSING")
        print("="*60)
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # Clean data
            df = self.clean_data(df)
            
            # Create binary labels
            df = self.create_binary_labels(df)
            
            # Balance dataset
            df = self.balance_dataset(df)
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X, y)
            
            # Split and scale data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_and_scale_data(X_seq, y_seq)
            
            # Save processed data
            self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
            
            print("\n" + "="*60)
            print("PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ Dataset processed and ready for training")
            print("✅ Balanced dataset to prevent class imbalance")
            print("✅ Proper train/validation/test split")
            print("✅ Feature scaling applied")
            print("✅ Sequences created for temporal modeling")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ Data directory not found.")
        print("Please run 'python download_cic_ids_dataset.py' first to download the dataset.")
        return
    
    # Initialize preprocessor
    preprocessor = CICIDSPreprocessor()
    
    # Process dataset
    success = preprocessor.process_dataset()
    
    if success:
        print("\n🎉 Ready for training!")
        print("Next step: Run 'python train_model.py' to train the CNN-TCN model.")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
