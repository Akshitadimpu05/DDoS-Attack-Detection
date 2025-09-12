#!/usr/bin/env python3
"""
Create a balanced synthetic dataset for CNN-TCN training
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def create_synthetic_normal_traffic(n_samples=5000, n_features=26, random_state=42):
    """Create synthetic normal network traffic patterns with realistic IoT characteristics"""
    np.random.seed(random_state)
    
    # Normal traffic characteristics for IoT devices
    normal_data = []
    
    for _ in range(n_samples):
        # Normal IoT traffic patterns:
        # - Periodic sensor readings
        # - Small packet sizes
        # - Regular intervals
        # - Low bandwidth usage
        
        # Base traffic features
        pkts = max(1, np.random.normal(25, 8))  # Small packet counts
        bytes_per_pkt = np.random.normal(64, 16)  # Small IoT packets
        total_bytes = pkts * bytes_per_pkt
        duration = np.random.exponential(1.5)  # Short connections
        
        # Calculate rates
        rate = total_bytes / max(duration, 0.1)
        srate = rate * np.random.uniform(0.4, 0.6)  # Balanced src/dst
        drate = rate * np.random.uniform(0.4, 0.6)
        
        # Protocol features (normal patterns)
        tcp_flags = np.random.choice([2, 18, 24], p=[0.6, 0.3, 0.1])  # SYN, PSH-ACK, FIN
        window_size = np.random.normal(8192, 1024)
        
        # Flow features
        fwd_pkts = pkts * np.random.uniform(0.45, 0.55)
        bwd_pkts = pkts - fwd_pkts
        
        # Timing features (regular patterns)
        iat_mean = duration / max(pkts, 1)
        iat_std = iat_mean * np.random.uniform(0.1, 0.3)  # Low variance
        
        # Statistical features
        pkt_len_mean = bytes_per_pkt
        pkt_len_std = bytes_per_pkt * np.random.uniform(0.1, 0.2)
        
        # Create feature vector
        features = [
            pkts, total_bytes, duration, rate, srate, drate,
            tcp_flags, window_size, fwd_pkts, bwd_pkts,
            iat_mean, iat_std, pkt_len_mean, pkt_len_std
        ]
        
        # Add remaining features with normal distributions
        while len(features) < n_features:
            features.append(np.random.normal(0, 0.5))
        
        normal_data.append(features[:n_features])
    
    return np.array(normal_data)

def create_synthetic_attack_traffic(n_samples=5000, n_features=26, random_state=42):
    """Create synthetic DDoS attack traffic patterns targeting IoT devices"""
    np.random.seed(random_state + 1)
    
    # DDoS attack characteristics
    attack_data = []
    
    for _ in range(n_samples):
        # Choose attack type
        attack_type = np.random.choice(['volumetric', 'protocol', 'application'], p=[0.5, 0.3, 0.2])
        
        if attack_type == 'volumetric':  # High volume DDoS
            pkts = max(100, np.random.normal(800, 300))  # Very high packet count
            bytes_per_pkt = np.random.normal(1024, 512)  # Medium packets
            duration = np.random.exponential(0.3)  # Very short bursts
            
        elif attack_type == 'protocol':  # SYN flood, etc.
            pkts = max(50, np.random.normal(400, 150))  # High packet count
            bytes_per_pkt = np.random.normal(40, 10)  # Small packets (headers only)
            duration = np.random.exponential(0.1)  # Very short
            
        else:  # Application layer attacks
            pkts = max(20, np.random.normal(150, 50))  # Moderate packet count
            bytes_per_pkt = np.random.normal(512, 256)  # Variable sizes
            duration = np.random.exponential(2.0)  # Longer connections
        
        total_bytes = pkts * bytes_per_pkt
        
        # Calculate abnormal rates
        rate = total_bytes / max(duration, 0.01)  # Very high rates
        srate = rate * np.random.uniform(0.8, 1.0)  # Source-heavy
        drate = rate * np.random.uniform(0.0, 0.2)  # Destination-light
        
        # Abnormal protocol features
        if attack_type == 'protocol':
            tcp_flags = 2  # Only SYN flags (SYN flood)
        else:
            tcp_flags = np.random.choice([2, 4, 16], p=[0.7, 0.2, 0.1])  # Abnormal flag patterns
        
        window_size = np.random.choice([0, 65535], p=[0.3, 0.7])  # Extreme values
        
        # Imbalanced flow
        fwd_pkts = pkts * np.random.uniform(0.8, 1.0)  # Source-heavy
        bwd_pkts = pkts - fwd_pkts
        
        # Irregular timing (burst patterns)
        iat_mean = duration / max(pkts, 1)
        iat_std = iat_mean * np.random.uniform(2.0, 5.0)  # High variance
        
        # Packet size patterns
        if attack_type == 'protocol':
            pkt_len_mean = bytes_per_pkt
            pkt_len_std = 0  # Very uniform (suspicious)
        else:
            pkt_len_mean = bytes_per_pkt
            pkt_len_std = bytes_per_pkt * np.random.uniform(0.5, 1.0)  # High variance
        
        # Create feature vector
        features = [
            pkts, total_bytes, duration, rate, srate, drate,
            tcp_flags, window_size, fwd_pkts, bwd_pkts,
            iat_mean, iat_std, pkt_len_mean, pkt_len_std
        ]
        
        # Add remaining features with attack distributions
        while len(features) < n_features:
            features.append(np.random.normal(2, 1.5))  # Shifted distribution
        
        attack_data.append(features[:n_features])
    
    return np.array(attack_data)

def create_time_windows(data, labels, window_size=50, stride=25):
    """Create time windows from the data"""
    windows = []
    window_labels = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        # Use majority label for the window
        window_label = np.round(np.mean(labels[i:i + window_size]))
        
        windows.append(window)
        window_labels.append(window_label)
    
    return np.array(windows), np.array(window_labels)

def create_balanced_dataset():
    """Create a balanced dataset for training"""
    print("Creating balanced synthetic dataset...")
    
    n_features = 26
    n_normal = 12000
    n_attack = 12000
    
    # Create synthetic data
    normal_data = create_synthetic_normal_traffic(n_normal, n_features)
    attack_data = create_synthetic_attack_traffic(n_attack, n_features)
    
    # Combine data
    X = np.vstack([normal_data, attack_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Created dataset: {X.shape}, Labels: {y.shape}")
    print(f"Normal samples: {(y == 0).sum()}, Attack samples: {(y == 1).sum()}")
    
    # Create time windows
    X_windows, y_windows = create_time_windows(X, y, window_size=50, stride=25)
    
    print(f"Time windows: {X_windows.shape}, Window labels: {y_windows.shape}")
    print(f"Normal windows: {(y_windows == 0).sum()}, Attack windows: {(y_windows == 1).sum()}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_windows, test_size=0.3, random_state=42, stratify=y_windows
    )
    
    # Normalize features
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
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
    joblib.dump(scaler, "processed_data/scaler.pkl")
    
    print("✅ Balanced dataset created and saved!")
    print(f"Training data: {X_train_scaled.shape}")
    print(f"Test data: {X_test_scaled.shape}")
    print(f"Training labels: Normal={np.sum(y_train == 0)}, Attack={np.sum(y_train == 1)}")
    print(f"Test labels: Normal={np.sum(y_test == 0)}, Attack={np.sum(y_test == 1)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    create_balanced_dataset()
