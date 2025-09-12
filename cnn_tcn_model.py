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
    """CNN-TCN Model for IoT DDoS Detection"""
    
    def __init__(self, input_shape, num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the CNN-TCN model architecture with improved regularization"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # CNN layers with optimized complexity
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                         kernel_regularizer=l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Temporal layers with dilated convolutions (TCN-like)
        x = layers.Conv1D(filters=32, kernel_size=3, dilation_rate=1, 
                         padding='causal', activation='relu',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=32, kernel_size=3, dilation_rate=2, 
                         padding='causal', activation='relu',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=32, kernel_size=3, dilation_rate=4, 
                         padding='causal', activation='relu',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Global pooling
        tcn_out = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with optimized regularization
        dense = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(tcn_out)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(0.2)(dense)
        
        dense = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense)
        dense = layers.Dropout(0.1)(dense)
        
        # Output layers for probability and quantiles
        # Main output: attack probability
        prob_output = layers.Dense(1, activation='sigmoid', name='attack_prob')(dense)
        
        # Quantile outputs for CRPS calculation
        q10_output = layers.Dense(1, activation='linear', name='q10')(dense)
        q50_output = layers.Dense(1, activation='linear', name='q50')(dense)
        q90_output = layers.Dense(1, activation='linear', name='q90')(dense)
        
        # Create model
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
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate losses and metrics"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Custom loss function combining binary crossentropy and quantile loss
        def combined_loss(y_true, y_pred):
            # Binary crossentropy for main classification
            bce_loss = keras.losses.binary_crossentropy(y_true, y_pred['attack_prob'])
            
            # Quantile loss for CRPS calculation
            q10_loss = self.quantile_loss(0.1)(y_true, y_pred['q10'])
            q50_loss = self.quantile_loss(0.5)(y_true, y_pred['q50'])
            q90_loss = self.quantile_loss(0.9)(y_true, y_pred['q90'])
            
            # Combine losses
            total_loss = bce_loss + 0.1 * (q10_loss + q50_loss + q90_loss)
            return total_loss
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'attack_prob': 'binary_crossentropy',
                'q10': self.quantile_loss(0.1),
                'q50': self.quantile_loss(0.5),
                'q90': self.quantile_loss(0.9)
            },
            loss_weights={
                'attack_prob': 1.0,
                'q10': 0.1,
                'q50': 0.1,
                'q90': 0.1
            },
            metrics={
                'attack_prob': ['accuracy', 'precision', 'recall'],
                'q10': ['mae'],
                'q50': ['mae'],
                'q90': ['mae']
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
    """CRPS (Continuous Ranked Probability Score) calculation utilities"""
    
    @staticmethod
    def calculate_crps_gaussian(observations, predictions, std_dev):
        """Calculate CRPS for Gaussian distribution"""
        import scipy.stats as stats
        
        # Normalize observations
        z = (observations - predictions) / std_dev
        
        # CRPS formula for Gaussian distribution
        crps = std_dev * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
        
        return crps
    
    @staticmethod
    def calculate_crps_quantiles(observations, q10, q50, q90):
        """Calculate CRPS from quantile predictions"""
        
        # Simple CRPS approximation using quantiles
        crps_scores = []
        
        for obs, q1, q5, q9 in zip(observations, q10, q50, q90):
            if obs <= q1:
                score = (q1 - obs) + 0.1 * (q5 - q1) + 0.4 * (q9 - q5)
            elif obs <= q5:
                score = 0.1 * (obs - q1) + 0.4 * (q9 - q5)
            elif obs <= q9:
                score = 0.4 * (obs - q5) + 0.1 * (q9 - obs)
            else:
                score = 0.4 * (q9 - q5) + (obs - q9)
            
            crps_scores.append(score)
        
        return np.array(crps_scores)
    
    @staticmethod
    def compute_global_threshold(crps_scores, percentile=95):
        """Compute global CRPS threshold"""
        return np.percentile(crps_scores, percentile)

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
