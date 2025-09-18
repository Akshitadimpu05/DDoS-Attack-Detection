#!/usr/bin/env python3
"""
Enhanced preprocessing script for NSL-KDD dataset
Handles the dataset with proper memory management and prevents overfitting
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

class NSLKDDPreprocessor:
    """Enhanced preprocessor for NSL-KDD dataset"""
    
    def __init__(self, data_dir="data", processed_dir="output_data"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Create processed data directory
        os.makedirs(processed_dir, exist_ok=True)
        
        # Dataset configuration
        self.target_column = 'attack_type'
        self.sequence_length = 10  # Reduced sequence length for better performance
        self.step_size = 1  # Smaller step size for more data
        
    def load_dataset(self):
        """Load NSL-KDD dataset with memory-efficient processing"""
        
        print("Loading NSL-KDD dataset...")
        
        # Try to load the main training file first
        train_file = os.path.join(self.data_dir, 'nsl_kdd_train.csv')
        test_file = os.path.join(self.data_dir, 'nsl_kdd_test.csv')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError("NSL-KDD CSV files not found. Please run 'python download_nsl_kdd_dataset.py' first.")
        
        # Load training data
        print("Loading training data...")
        df_train = pd.read_csv(train_file)
        print(f"Training data loaded: {len(df_train):,} records")
        
        # Load test data if available
        if os.path.exists(test_file):
            print("Loading test data...")
            df_test = pd.read_csv(test_file)
            print(f"Test data loaded: {len(df_test):,} records")
            
            # Combine for preprocessing
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            df = df_train
        
        print(f"Total dataset size: {len(df):,} records")
        return df
    
    def clean_data(self, df):
        """Clean and prepare the NSL-KDD dataset"""
        
        print("Cleaning NSL-KDD dataset...")
        
        # Remove difficulty_level column if present (not needed for classification)
        if 'difficulty_level' in df.columns:
            df = df.drop('difficulty_level', axis=1)
        
        # Handle missing values
        df = df.dropna()
        
        # Remove duplicate rows
        initial_size = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_size - len(df):,} duplicate records")
        
        print(f"Dataset shape after cleaning: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features using label encoding"""
        
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            if col in df.columns:
                print(f"  Encoding {col}...")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def create_binary_labels(self, df):
        """Convert multi-class attack types to binary (Normal vs Attack)"""
        
        print("Creating binary labels...")
        
        # Get unique attack types
        unique_attacks = df[self.target_column].unique()
        print(f"Unique attack types found: {len(unique_attacks)}")
        print(f"Attack types: {list(unique_attacks)}")
        
        # Create binary labels (0: Normal, 1: Attack)
        df['binary_label'] = 0  # Default to normal
        
        # Mark attacks as 1
        normal_labels = ['normal']
        for attack in unique_attacks:
            if attack.lower() not in normal_labels:
                df.loc[df[self.target_column] == attack, 'binary_label'] = 1
        
        # Count distribution
        label_counts = df['binary_label'].value_counts()
        print(f"Label distribution:")
        print(f"  Normal (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Attack (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def balance_dataset(self, df, max_samples_per_class=50000):
        """Balance the dataset with much larger sample size for better training"""
        
        print("Balancing dataset with increased sample size...")
        
        # Separate classes
        normal_data = df[df['binary_label'] == 0]
        attack_data = df[df['binary_label'] == 1]
        
        print(f"Before balancing:")
        print(f"  Normal samples: {len(normal_data):,}")
        print(f"  Attack samples: {len(attack_data):,}")
        
        # Use much larger sample sizes for better training
        normal_samples = min(len(normal_data), max_samples_per_class)
        attack_samples = min(len(attack_data), max_samples_per_class)
        
        # Take larger balanced samples
        normal_balanced = normal_data.sample(n=normal_samples, random_state=42)
        attack_balanced = attack_data.sample(n=attack_samples, random_state=42)
        
        # Combine balanced data
        balanced_df = pd.concat([normal_balanced, attack_balanced], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After balancing:")
        print(f"  Total samples: {len(balanced_df):,}")
        print(f"  Normal samples: {len(balanced_df[balanced_df['binary_label'] == 0]):,}")
        print(f"  Attack samples: {len(balanced_df[balanced_df['binary_label'] == 1]):,}")
        
        return balanced_df
    
    def prepare_features(self, df):
        """Prepare features with zero variance removal"""
        
        print("Preparing features...")
        
        # Select numeric features only (exclude target columns)
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in [self.target_column, 'binary_label']]
        
        print(f"Initial features: {len(feature_columns)}")
        
        # Extract features and labels
        X = df[feature_columns].values
        y = df['binary_label'].values
        
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Remove zero variance features
        feature_variances = np.var(X, axis=0)
        non_zero_var_indices = feature_variances > 1e-8
        
        X = X[:, non_zero_var_indices]
        feature_columns = [feature_columns[i] for i in range(len(feature_columns)) if non_zero_var_indices[i]]
        
        removed_features = np.sum(~non_zero_var_indices)
        print(f"Removed {removed_features} zero variance features")
        print(f"Final features: {len(feature_columns)}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y, feature_columns
    
    def create_sequences(self, X, y):
        """Create sequences for CNN-TCN model with proper sliding window"""
        
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
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(self.processed_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.processed_dir, 'label_encoders.pkl'))
        
        # Save feature names and metadata
        metadata = {
            'dataset_name': 'NSL-KDD',
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
        print("NSL-KDD DATASET PREPROCESSING")
        print("="*60)
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # Clean data
            df = self.clean_data(df)
            
            # Encode categorical features
            df = self.encode_categorical_features(df)
            
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
            print("✅ NSL-KDD dataset processed and ready for training")
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
        print("Please run 'python download_nsl_kdd_dataset.py' first to download the dataset.")
        return
    
    # Initialize preprocessor
    preprocessor = NSLKDDPreprocessor()
    
    # Process dataset
    success = preprocessor.process_dataset()
    
    if success:
        print("\n🎉 Ready for training!")
        print("Next step: Run 'python train_enhanced_model.py' to train the CNN-TCN model.")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
