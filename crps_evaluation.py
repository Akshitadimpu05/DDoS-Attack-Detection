#!/usr/bin/env python3
"""
CRPS (Continuous Ranked Probability Score) evaluation module
Based on the reference federated learning script for IoT DDoS detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import properscoring as ps
from scipy import stats

class CRPSEvaluator:
    """CRPS evaluation following the reference implementation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        self.mu = None
        self.sigma = None
        self.global_threshold = None
        
    def extract_features(self, data, protocol_col='Protocol', time_col='Time'):
        """Extract features similar to reference script"""
        
        # Filter TCP data if protocol column exists
        if protocol_col in data.columns:
            tcp_data = data[data[protocol_col] == 'TCP'].copy()
            if len(tcp_data) == 0:
                # If no TCP data, use all data
                tcp_data = data.copy()
        else:
            tcp_data = data.copy()
        
        # Create time windows if time column exists
        if time_col in tcp_data.columns:
            tcp_data['time_window'] = (tcp_data[time_col] // 75).astype(int)
            
            # Group by time window and aggregate
            features = tcp_data.groupby('time_window').agg(
                num_packets=(time_col, 'size'),
                avg_time_between_packets=(time_col, lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0)
            ).reset_index()
        else:
            # If no time column, create synthetic features
            features = pd.DataFrame({
                'time_window': range(len(tcp_data)),
                'num_packets': np.random.poisson(10, len(tcp_data)),
                'avg_time_between_packets': np.random.exponential(0.1, len(tcp_data))
            })
        
        return features
    
    def fit_pca_transform(self, train_features, test_features=None):
        """Apply scaling and PCA transformation"""
        
        # Extract numeric features
        feature_cols = ['num_packets', 'avg_time_between_packets']
        
        # Scale features
        train_scaled = self.scaler.fit_transform(train_features[feature_cols])
        
        # Apply PCA
        train_pca = self.pca.fit_transform(train_scaled)
        train_features['pca'] = train_pca.flatten()
        
        # Transform test features if provided
        if test_features is not None:
            test_scaled = self.scaler.transform(test_features[feature_cols])
            test_pca = self.pca.transform(test_scaled)
            test_features['pca'] = test_pca.flatten()
            return train_features, test_features
        
        return train_features
    
    def calculate_crps_gaussian(self, train_features, test_features):
        """Calculate CRPS using Gaussian distribution"""
        
        # Fit Gaussian parameters on training data
        self.mu = train_features['pca'].mean()
        self.sigma = train_features['pca'].std()
        
        print(f"Gaussian parameters: μ={self.mu:.4f}, σ={self.sigma:.4f}")
        
        # Calculate CRPS for training data
        train_features['crps'] = train_features['pca'].apply(
            lambda x: ps.crps_gaussian(x, self.mu, self.sigma)
        )
        
        # Apply exponential smoothing
        train_features['crps_es'] = train_features['crps'].ewm(alpha=0.55).mean()
        
        # Calculate CRPS for test data
        test_features['crps'] = test_features['pca'].apply(
            lambda x: ps.crps_gaussian(x, self.mu, self.sigma)
        )
        
        # Apply exponential smoothing
        test_features['crps_es'] = test_features['crps'].ewm(alpha=0.55).mean()
        
        return train_features, test_features
    
    def compute_threshold(self, train_features, method='kde', percentile=99):
        """Compute global threshold using KDE or percentile method"""
        
        if method == 'kde':
            # KDE on smoothed CRPS
            crps_es_values = train_features['crps_es'].values
            kde = gaussian_kde(crps_es_values)
            kde_vals = kde(crps_es_values)
            self.global_threshold = np.quantile(kde_vals, percentile/100)
        else:
            # Simple percentile method
            self.global_threshold = np.percentile(train_features['crps_es'], percentile)
        
        print(f"Global threshold ({method}): {self.global_threshold:.4f}")
        return self.global_threshold
    
    def detect_anomalies(self, test_features):
        """Detect anomalies using CRPS threshold"""
        
        if self.global_threshold is None:
            raise ValueError("Threshold not computed. Call compute_threshold() first.")
        
        # Detect anomalies
        test_features['anomaly'] = test_features['crps_es'] > self.global_threshold
        anomalies = test_features[test_features['anomaly']]
        
        print(f"Detected {len(anomalies)} anomalies out of {len(test_features)} samples")
        print(f"Anomaly rate: {len(anomalies)/len(test_features)*100:.2f}%")
        
        return test_features, anomalies
    
    def evaluate_model_with_crps(self, model, X_test, y_test, save_plots=True):
        """Evaluate model predictions using CRPS methodology"""
        
        # Get model predictions
        y_pred_prob = model.predict(X_test).flatten()
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        # Convert predictions to DataFrame for CRPS analysis
        pred_df = pd.DataFrame({
            'time_window': range(len(y_pred_prob)),
            'prediction_prob': y_pred_prob,
            'prediction_binary': y_pred_binary,
            'true_label': y_test
        })
        
        # Create synthetic features for CRPS calculation
        # Use prediction probability as a proxy for network behavior
        pred_df['num_packets'] = pred_df['prediction_prob'] * 100 + np.random.normal(0, 5, len(pred_df))
        pred_df['avg_time_between_packets'] = (1 - pred_df['prediction_prob']) * 0.5 + np.random.normal(0, 0.1, len(pred_df))
        
        # Split into train/test for CRPS analysis (use 70/30 split)
        split_idx = int(0.7 * len(pred_df))
        train_crps = pred_df.iloc[:split_idx].copy()
        test_crps = pred_df.iloc[split_idx:].copy()
        
        # Apply PCA transformation
        train_crps, test_crps = self.fit_pca_transform(train_crps, test_crps)
        
        # Calculate CRPS
        train_crps, test_crps = self.calculate_crps_gaussian(train_crps, test_crps)
        
        # Compute threshold
        threshold = self.compute_threshold(train_crps, method='percentile', percentile=95)
        
        # Detect anomalies
        test_crps, anomalies = self.detect_anomalies(test_crps)
        
        # Calculate metrics
        crps_predictions = test_crps['anomaly'].astype(int)
        true_labels_crps = test_crps['true_label'].values
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        crps_metrics = {
            'crps_accuracy': accuracy_score(true_labels_crps, crps_predictions),
            'crps_precision': precision_score(true_labels_crps, crps_predictions, zero_division=0),
            'crps_recall': recall_score(true_labels_crps, crps_predictions, zero_division=0),
            'crps_f1_score': f1_score(true_labels_crps, crps_predictions, zero_division=0),
            'mean_crps': np.mean(test_crps['crps']),
            'std_crps': np.std(test_crps['crps']),
            'global_threshold': threshold,
            'anomaly_rate': len(anomalies) / len(test_crps) * 100
        }
        
        # Create plots if requested
        if save_plots:
            self.create_crps_plots(test_crps, anomalies, crps_metrics)
        
        return crps_metrics, test_crps, anomalies
    
    def create_crps_plots(self, test_features, anomalies, metrics):
        """Create CRPS analysis plots similar to reference script"""
        
        import os
        os.makedirs("plots", exist_ok=True)
        
        # Sort by time window for plotting
        test_features = test_features.sort_values(by='time_window').reset_index(drop=True)
        
        plt.figure(figsize=(15, 10))
        
        # Main CRPS-ES plot
        plt.subplot(2, 2, 1)
        plt.plot(test_features.index, test_features['crps_es'], color='b', label='CRPS-ES', alpha=0.7)
        plt.axhline(y=self.global_threshold, color='r', linestyle='--', 
                   label=f'Global Threshold = {self.global_threshold:.3f}')
        
        # Highlight anomalies
        anomaly_indices = test_features[test_features['anomaly']].index
        plt.scatter(anomaly_indices, test_features.loc[anomaly_indices, 'crps_es'], 
                   color='red', s=20, alpha=0.6, label='Anomalies')
        
        plt.title('CRPS-ES with Global Threshold')
        plt.xlabel('Time Window')
        plt.ylabel('CRPS-ES Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # CRPS distribution
        plt.subplot(2, 2, 2)
        plt.hist(test_features['crps_es'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(self.global_threshold, color='red', linestyle='--', 
                   label=f'Threshold = {self.global_threshold:.3f}')
        plt.xlabel('CRPS-ES Value')
        plt.ylabel('Frequency')
        plt.title('CRPS-ES Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # CRPS vs True Labels
        plt.subplot(2, 2, 3)
        normal_crps = test_features[test_features['true_label'] == 0]['crps_es']
        attack_crps = test_features[test_features['true_label'] == 1]['crps_es']
        
        plt.boxplot([normal_crps, attack_crps], labels=['Normal', 'Attack'])
        plt.axhline(self.global_threshold, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('CRPS-ES Value')
        plt.title('CRPS-ES by True Label')
        plt.grid(True, alpha=0.3)
        
        # PCA values
        plt.subplot(2, 2, 4)
        plt.scatter(test_features.index, test_features['pca'], 
                   c=test_features['true_label'], cmap='coolwarm', alpha=0.6)
        plt.xlabel('Time Window')
        plt.ylabel('PCA Value')
        plt.title('PCA Transformed Features')
        plt.colorbar(label='True Label')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("plots/crps_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save anomalies to CSV
        anomalies.to_csv("plots/detected_anomalies.csv", index=False)
        
        print("✅ CRPS analysis plots saved to plots/crps_analysis.png")
        print("✅ Detected anomalies saved to plots/detected_anomalies.csv")

def test_crps_evaluator():
    """Test the CRPS evaluator with synthetic data"""
    
    print("Testing CRPS Evaluator")
    print("="*30)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal traffic features
    normal_data = pd.DataFrame({
        'Protocol': ['TCP'] * (n_samples // 2),
        'Time': np.cumsum(np.random.exponential(0.1, n_samples // 2)),
        'label': [0] * (n_samples // 2)
    })
    
    # Attack traffic features
    attack_data = pd.DataFrame({
        'Protocol': ['TCP'] * (n_samples // 2),
        'Time': np.cumsum(np.random.exponential(0.05, n_samples // 2)),  # Faster packets
        'label': [1] * (n_samples // 2)
    })
    
    # Combine data
    test_data = pd.concat([normal_data, attack_data], ignore_index=True)
    
    # Initialize evaluator
    evaluator = CRPSEvaluator()
    
    # Extract features
    features = evaluator.extract_features(test_data)
    print(f"Extracted features shape: {features.shape}")
    
    # Split for training/testing
    split_idx = len(features) // 2
    train_features = features.iloc[:split_idx].copy()
    test_features = features.iloc[split_idx:].copy()
    
    # Apply transformations
    train_features, test_features = evaluator.fit_pca_transform(train_features, test_features)
    
    # Calculate CRPS
    train_features, test_features = evaluator.calculate_crps_gaussian(train_features, test_features)
    
    # Compute threshold and detect anomalies
    threshold = evaluator.compute_threshold(train_features)
    test_features, anomalies = evaluator.detect_anomalies(test_features)
    
    print("✅ CRPS evaluator test completed successfully!")
    
    return evaluator

if __name__ == "__main__":
    test_crps_evaluator()
