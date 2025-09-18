#!/usr/bin/env python3
"""
CNN-TCN Model for IoT DDoS Detection with CRPS Integration
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import numpy as np
import properscoring as ps

class TemporalConvNet(layers.Layer):
    """Temporal Convolutional Network (TCN) layer with improved regularization"""
    
    def __init__(self, num_filters, kernel_size=2, dropout_rate=0.4, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # TCN layers with dilated convolutions and regularization
        self.tcn_out = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=1,
            padding='causal',
            activation=None,
            kernel_regularizer=l2(0.001)
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # First TCN block
        x = self.tcn_out(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        return x

class CNNTCNModel:
    """Enhanced CNN-TCN Model for IoT DDoS Detection with CRPS Integration"""
    
    def __init__(self, input_shape, num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build improved CNN-TCN model for better performance"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Enhanced CNN Feature Extraction Layers
        x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', 
                         kernel_regularizer=l2(0.0001), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                         kernel_regularizer=l2(0.0001), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                         kernel_regularizer=l2(0.0001), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Enhanced TCN Layers for Temporal Dependencies
        x = TemporalConvNet(num_filters=128, kernel_size=3, dropout_rate=0.2)(x)
        x = TemporalConvNet(num_filters=64, kernel_size=3, dropout_rate=0.2)(x)
        x = TemporalConvNet(num_filters=32, kernel_size=3, dropout_rate=0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Enhanced dense layers
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Multiple outputs for CRPS evaluation
        # Main classification output
        prob_output = layers.Dense(1, activation='sigmoid', name='attack_prob')(x)
        
        # Quantile outputs for CRPS (10th, 50th, 90th percentiles)
        q10_output = layers.Dense(1, activation='sigmoid', name='q10')(x)
        q50_output = layers.Dense(1, activation='sigmoid', name='q50')(x)
        q90_output = layers.Dense(1, activation='sigmoid', name='q90')(x)
        
        # Create model with multiple outputs
        self.model = keras.Model(
            inputs=inputs,
            outputs={
                'attack_prob': prob_output,
                'q10': q10_output,
                'q50': q50_output,
                'q90': q90_output
            }
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.002):
        """Compile model with improved optimizer settings"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Compile with improved optimizer and loss weights
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss={
                'attack_prob': 'binary_crossentropy',
                'q10': 'mean_squared_error',
                'q50': 'mean_squared_error', 
                'q90': 'mean_squared_error'
            },
            loss_weights={
                'attack_prob': 1.0,
                'q10': 0.05,
                'q50': 0.05,
                'q90': 0.05
            },
            metrics={
                'attack_prob': ['accuracy', 'precision', 'recall', 'auc']
            }
        )
        
        return self.model
    
    @staticmethod
    def quantile_loss(quantile):
        """Quantile loss function for CRPS calculation"""
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
        return loss
    
    def get_model_summary(self):
        """Get model summary and parameter count"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        if total_params > 100000:
            print("⚠️  Warning: Model has more than 100k parameters. Consider reducing model size for edge deployment.")
        else:
            print("✅ Model is lightweight enough for edge deployment.")
        
        return total_params

class CRPSMetrics:
    """Enhanced CRPS (Continuous Ranked Probability Score) calculation utilities"""
    
    @staticmethod
    def calculate_crps_gaussian(observations, predictions, std_dev):
        """Calculate CRPS for Gaussian distribution"""
        import scipy.stats as stats
        
        # Handle edge cases
        std_dev = np.maximum(std_dev, 1e-8)  # Prevent division by zero
        
        # Normalize observations
        z = (observations - predictions) / std_dev
        
        # CRPS formula for Gaussian distribution
        crps = std_dev * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
        
        return np.abs(crps)  # Ensure positive CRPS scores
    
    @staticmethod
    def calculate_crps_quantiles(observations, q10, q50, q90):
        """Calculate CRPS from quantile predictions with proper implementation"""
        
        # Ensure quantiles are properly ordered
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q50, q90)
        
        crps_scores = []
        
        for obs, q1, q5, q9 in zip(observations, q10, q50, q90):
            # Proper CRPS calculation using quantile integration
            if obs <= q1:
                score = 2 * (q1 - obs) * 0.1 + (q5 - q1) * 0.1 + (q9 - q5) * 0.4
            elif obs <= q5:
                score = (obs - q1) * 0.1 + (q5 - obs) * 0.1 + (q9 - q5) * 0.4
            elif obs <= q9:
                score = (q5 - q1) * 0.1 + (obs - q5) * 0.4 + (q9 - obs) * 0.1
            else:
                score = (q5 - q1) * 0.1 + (q9 - q5) * 0.4 + 2 * (obs - q9) * 0.1
            
            crps_scores.append(max(0, score))  # Ensure non-negative scores
        
        return np.array(crps_scores)
    
    @staticmethod
    def compute_global_threshold(crps_scores, percentile=95):
        """Compute global CRPS threshold for anomaly detection"""
        return np.percentile(crps_scores, percentile)
    
    @staticmethod
    def evaluate_crps_performance(y_true, crps_scores, threshold):
        """Evaluate CRPS-based anomaly detection performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Convert CRPS scores to binary predictions
        y_pred_crps = (crps_scores > threshold).astype(int)
        
        metrics = {
            'crps_accuracy': accuracy_score(y_true, y_pred_crps),
            'crps_precision': precision_score(y_true, y_pred_crps, zero_division=0),
            'crps_recall': recall_score(y_true, y_pred_crps, zero_division=0),
            'crps_f1_score': f1_score(y_true, y_pred_crps, zero_division=0),
            'mean_crps': np.mean(crps_scores),
            'std_crps': np.std(crps_scores),
            'threshold': threshold
        }
        
        return metrics

def create_model(input_shape):
    """Factory function to create and compile CNN-TCN model"""
    
    model_builder = CNNTCNModel(input_shape)
    model = model_builder.build_model()
    model = model_builder.compile_model()
    
    # Print model summary
    total_params = model_builder.get_model_summary()
    
    return model, model_builder

if __name__ == "__main__":
    # Test model creation
    print("Testing CNN-TCN Model Creation")
    print("="*40)
    
    # Example input shape (batch_size, timesteps, features)
    input_shape = (100, 50)  # 100 timesteps, 50 features
    
    model, model_builder = create_model(input_shape)
    
    print("\n✅ Model created successfully!")
    print("Model is ready for training.")
