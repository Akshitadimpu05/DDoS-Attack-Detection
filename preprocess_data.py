#!/usr/bin/env python3
"""
Data preprocessing for Bot-IoT dataset
Extracts flow features and prepares data for CNN-TCN training
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BotIoTPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load Bot-IoT dataset from multiple CSV files"""
        print("Loading Bot-IoT dataset...")
        
        # Find all CSV files except data_names.csv
        csv_files = [f for f in os.listdir(self.data_dir) 
                    if f.endswith('.csv') and 'names' not in f.lower()]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data directory")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load and combine multiple files to get diverse data
        dataframes = []
        normal_count = 0
        attack_count = 0
        
        # Shuffle files for better diversity
        import random
        random.seed(42)
        shuffled_files = csv_files.copy()
        random.shuffle(shuffled_files)
        
        # First, load data_1.csv which has more normal traffic
        priority_files = ['data_1.csv'] + [f for f in shuffled_files if f != 'data_1.csv']
        
        for csv_file in priority_files[:20]:  # Load more files for diversity
            try:
                file_path = os.path.join(self.data_dir, csv_file)
                df_temp = pd.read_csv(file_path, low_memory=False)
                
                # Check what type of data this file contains
                if 'attack' in df_temp.columns:
                    file_normal = (df_temp['attack'] == 0).sum()
                    file_attack = (df_temp['attack'] == 1).sum()
                    
                    print(f"  {csv_file}: {len(df_temp)} rows, Normal: {file_normal}, Attack: {file_attack}")
                    
                    # Balance the dataset - prioritize normal samples
                    if file_normal > 0 and normal_count < 15000:
                        # Take all available normal samples
                        normal_sample = df_temp[df_temp['attack'] == 0]
                        dataframes.append(normal_sample)
                        normal_count += len(normal_sample)
                    
                    if file_attack > 0 and attack_count < 15000:
                        # Take fewer attack samples to balance
                        attack_sample = df_temp[df_temp['attack'] == 1].sample(
                            min(1000, file_attack, 15000 - attack_count), random_state=42
                        )
                        dataframes.append(attack_sample)
                        attack_count += len(attack_sample)
                
            except Exception as e:
                print(f"  Warning: Could not load {csv_file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data could be loaded from CSV files")
        
        # Combine all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        print(f"Combined dataset shape: {df.shape}")
        print(f"Normal samples: {normal_count}, Attack samples: {attack_count}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def filter_relevant_classes(self, df):
        """Filter dataset to include only DDoS/DoS and normal traffic"""
        print("Filtering relevant classes...")
        
        # The Bot-IoT dataset uses 'attack' column: 0=Normal, 1=Attack
        # and 'category' column for attack types
        if 'attack' not in df.columns:
            print("Warning: No 'attack' column found. Using all data as is.")
            return df
        
        print(f"Attack column values: {df['attack'].unique()}")
        
        if 'category' in df.columns:
            print(f"Category values: {df['category'].unique()}")
            
            # Filter for Normal, DoS, and DDoS traffic only
            relevant_categories = []
            for category in df['category'].unique():
                cat_lower = str(category).lower()
                if any(keyword in cat_lower for keyword in ['normal', 'dos', 'ddos']):
                    relevant_categories.append(category)
            
            print(f"Relevant categories: {relevant_categories}")
            
            if relevant_categories:
                filtered_df = df[df['category'].isin(relevant_categories)].copy()
            else:
                # If no relevant categories found, use all data
                filtered_df = df.copy()
        else:
            # If no category column, use all data (attack=0 is normal, attack=1 is attack)
            filtered_df = df.copy()
        
        print(f"Filtered dataset shape: {filtered_df.shape}")
        
        # Show distribution
        if 'attack' in filtered_df.columns:
            attack_dist = filtered_df['attack'].value_counts()
            print(f"Attack distribution: {dict(attack_dist)}")
        
        return filtered_df
    
    def extract_flow_features(self, df):
        """Extract flow-based features from network traffic"""
        print("Extracting flow features...")
        
        features = {}
        
        # Basic packet statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if 'time' in col.lower() or 'duration' in col.lower():
                features[f'{col}_duration'] = df[col]
            elif 'packet' in col.lower() or 'pkt' in col.lower():
                features[f'{col}_count'] = df[col]
            elif 'byte' in col.lower() or 'size' in col.lower():
                features[f'{col}_bytes'] = df[col]
            elif 'flag' in col.lower():
                features[f'{col}_flags'] = df[col]
            else:
                features[col] = df[col]
        
        # Create additional engineered features
        if 'Bytes' in df.columns and 'Packets' in df.columns:
            features['bytes_per_packet'] = df['Bytes'] / (df['Packets'] + 1e-8)
        
        if 'Duration' in df.columns and 'Packets' in df.columns:
            features['packets_per_second'] = df['Packets'] / (df['Duration'] + 1e-8)
        
        # Protocol encoding
        if 'Protocol' in df.columns:
            protocol_encoded = pd.get_dummies(df['Protocol'], prefix='protocol')
            for col in protocol_encoded.columns:
                features[col] = protocol_encoded[col]
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(features)
        
        # Add label
        label_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['label', 'attack', 'category', 'class'])]
        
        if label_cols:
            label_col = label_cols[0]
            # Binary classification: 1 for attack, 0 for normal
            feature_df['label'] = df[label_col].apply(
                lambda x: 0 if 'normal' in str(x).lower() or 'benign' in str(x).lower() else 1
            )
        else:
            # If no label found, create dummy labels
            feature_df['label'] = 0
        
        print(f"Extracted features shape: {feature_df.shape}")
        print(f"Feature columns: {list(feature_df.columns)}")
        
        return feature_df
    
    def create_time_windows(self, df, window_size=50):
        """Create time windows for sequence modeling with better diversity"""
        print(f"Creating time windows of size {window_size}...")
        
        # Sort by time if time column exists
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols:
            df = df.sort_values(time_cols[0]).reset_index(drop=True)
        
        # Shuffle the dataframe to increase diversity
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create windows with smaller stride for more diversity
        windows = []
        labels = []
        
        stride = window_size // 4  # Smaller stride for more overlapping windows
        
        for i in tqdm(range(0, len(df) - window_size + 1, stride)):
            window = df.iloc[i:i + window_size]
            
            # Use majority label for the window, but add some noise tolerance
            label_counts = window['label'].value_counts()
            if len(label_counts) > 0:
                # If the window has mixed labels, use the majority but with threshold
                majority_label = label_counts.index[0]
                majority_ratio = label_counts.iloc[0] / len(window)
                
                # Only use windows with clear majority (>60%) to reduce noise
                if majority_ratio >= 0.6:
                    window_label = majority_label
                else:
                    continue  # Skip ambiguous windows
            else:
                continue
            
            # Extract features (excluding label)
            feature_cols = [col for col in df.columns if col != 'label']
            window_features = window[feature_cols].values
            
            windows.append(window_features)
            labels.append(window_label)
        
        return np.array(windows), np.array(labels)
    
    def preprocess_data(self, window_size=50, test_size=0.3):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Filter relevant classes
        df = self.filter_relevant_classes(df)
        
        # Extract features
        feature_df = self.extract_flow_features(df)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        # Create time windows
        X, y = self.create_time_windows(feature_df, window_size)
        
        print(f"Final data shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalize features
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
        
        # Save processed data
        os.makedirs("processed_data", exist_ok=True)
        
        np.save("processed_data/X_train.npy", X_train_scaled)
        np.save("processed_data/X_test.npy", X_test_scaled)
        np.save("processed_data/y_train.npy", y_train)
        np.save("processed_data/y_test.npy", y_test)
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, "processed_data/scaler.pkl")
        
        print("✅ Data preprocessing complete!")
        print(f"Training data: {X_train_scaled.shape}")
        print(f"Test data: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_data_visualization(self, df):
        """Create visualizations of the dataset"""
        print("Creating data visualizations...")
        
        os.makedirs("plots", exist_ok=True)
        
        # Label distribution
        plt.figure(figsize=(10, 6))
        label_counts = df['label'].value_counts()
        plt.bar(['Normal', 'Attack'], label_counts.values)
        plt.title('Label Distribution')
        plt.ylabel('Count')
        plt.savefig('plots/label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature correlation heatmap (sample of features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # First 20 numeric columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('plots/feature_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved to plots/ directory")

if __name__ == "__main__":
    print("Bot-IoT Data Preprocessor")
    print("="*30)
    
    preprocessor = BotIoTPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
        print("\n✅ Preprocessing complete!")
        print("Next step: Run 'python train_model.py' to train the CNN-TCN model.")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        print("Please ensure the dataset is downloaded and accessible.")
