#!/usr/bin/env python3
"""
CNN model with CRPS metrics for DDoS detection
Simplified architecture for high performance with uncertainty quantification
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

class CNNCRPSModel:
    """CNN model with CRPS uncertainty quantification for DDoS detection"""
    
    def __init__(self, input_shape=(10, 40), model_save_path="models", plots_save_path="plots"):
        self.input_shape = input_shape
        self.model_save_path = model_save_path
        self.plots_save_path = plots_save_path
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(plots_save_path, exist_ok=True)
    
    def crps_loss(self, y_true, y_pred):
        """CRPS loss function for uncertainty quantification"""
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    def quantile_loss(self, quantile):
        """Quantile loss for CRPS calculation"""
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
        return loss
    
    def crps_metric(self, y_true, y_pred_quantiles):
        """Calculate CRPS metric from quantiles"""
        q10, q50, q90 = y_pred_quantiles[:, 0], y_pred_quantiles[:, 1], y_pred_quantiles[:, 2]
        
        # CRPS calculation
        crps = tf.reduce_mean(
            (q90 - q10) * (0.9 - 0.1) / 2 +
            tf.maximum(q10 - y_true, 0) * 0.1 +
            tf.maximum(y_true - q90, 0) * 0.1 +
            tf.abs(q50 - y_true)
        )
        return crps
    
    def build_model(self):
        """Build CNN model with CRPS outputs"""
        
        print("Building CNN model with CRPS metrics...")
        
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # CNN layers for temporal feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        
        # Multiple outputs for CRPS
        attack_prob = layers.Dense(1, activation='sigmoid', name='attack_prob')(x)
        q10_output = layers.Dense(1, activation='sigmoid', name='q10')(x)
        q50_output = layers.Dense(1, activation='sigmoid', name='q50')(x)
        q90_output = layers.Dense(1, activation='sigmoid', name='q90')(x)
        
        self.model = keras.Model(inputs=inputs, outputs={
            'attack_prob': attack_prob,
            'q10': q10_output,
            'q50': q50_output,
            'q90': q90_output
        })
        
        # Compile with multiple losses
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss={
                'attack_prob': 'binary_crossentropy',
                'q10': self.quantile_loss(0.1),
                'q50': self.quantile_loss(0.5),
                'q90': self.quantile_loss(0.9)
            },
            loss_weights={
                'attack_prob': 1.0,
                'q10': 0.3,
                'q50': 0.3,
                'q90': 0.3
            },
            metrics={
                'attack_prob': ['accuracy', 'precision', 'recall'],
                'q10': ['mae'],
                'q50': ['mae'],
                'q90': ['mae']
            }
        )
        
        print("✅ CNN-CRPS model built successfully")
        print(f"Model parameters: {self.model.count_params():,}")
        
        return self.model
    
    def load_data(self):
        """Load processed sequence data"""
        
        print("Loading processed sequence data...")
        
        try:
            # Check if files exist in output_data, otherwise fall back to processed_data
            if os.path.exists(os.path.join('output_data', 'X_train.npy')):
                data_path = 'output_data'
            else:
                data_path = 'processed_data'
                
            X_train = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True)
            X_val = np.load(os.path.join(data_path, 'X_val.npy'), allow_pickle=True)
            X_test = np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
            y_train = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
            y_val = np.load(os.path.join(data_path, 'y_val.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
            
            # Prepare multi-output labels for CRPS
            y_train_dict = {
                'attack_prob': y_train,
                'q10': y_train,
                'q50': y_train,
                'q90': y_train
            }
            
            y_val_dict = {
                'attack_prob': y_val,
                'q10': y_val,
                'q50': y_val,
                'q90': y_val
            }
            
            print(f"✅ Data loaded:")
            print(f"  Training: {X_train.shape}")
            print(f"  Validation: {X_val.shape}")
            print(f"  Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train_dict, y_val_dict, y_test
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None, None, None, None
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=40, batch_size=64):
        """Train the model with early stopping"""
        
        print(f"Training CNN-CRPS model for maximum {epochs} epochs...")
        
        # Callbacks with early stopping
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_attack_prob_accuracy', 
                patience=8, 
                restore_best_weights=True, 
                verbose=1,
                mode='max',
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1,
                mode='min',
                min_delta=0.001
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_cnn_crps_model.h5'),
                monitor='val_attack_prob_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with CRPS metrics"""
        
        print("Evaluating CNN-CRPS model...")
        
        # Predictions
        predictions = self.model.predict(X_test, verbose=0)
        y_pred_prob = predictions['attack_prob'].flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Standard metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_prob)
        }
        
        # CRPS metrics
        q10_pred = predictions['q10'].flatten()
        q50_pred = predictions['q50'].flatten()
        q90_pred = predictions['q90'].flatten()
        
        # Calculate CRPS
        crps_scores = []
        for i in range(len(y_test)):
            quantiles = np.array([q10_pred[i], q50_pred[i], q90_pred[i]])
            crps = self.calculate_crps_single(y_test[i], quantiles)
            crps_scores.append(crps)
        
        crps_scores = np.array(crps_scores)
        
        # CRPS-based classification using median
        crps_threshold = np.median(crps_scores)
        y_pred_crps = (crps_scores < crps_threshold).astype(int)
        
        crps_metrics = {
            'crps_accuracy': accuracy_score(y_test, y_pred_crps),
            'crps_precision': precision_score(y_test, y_pred_crps),
            'crps_recall': recall_score(y_test, y_pred_crps),
            'crps_f1': f1_score(y_test, y_pred_crps),
            'mean_crps': np.mean(crps_scores),
            'std_crps': np.std(crps_scores),
            'crps_threshold': crps_threshold
        }
        
        print("\n" + "="*60)
        print("CNN-CRPS MODEL EVALUATION RESULTS")
        print("="*60)
        print("🎯 Standard Classification Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print("\n📊 CRPS-Enhanced Metrics:")
        print(f"  CRPS Accuracy: {crps_metrics['crps_accuracy']:.4f}")
        print(f"  CRPS Precision: {crps_metrics['crps_precision']:.4f}")
        print(f"  CRPS Recall: {crps_metrics['crps_recall']:.4f}")
        print(f"  CRPS F1-Score: {crps_metrics['crps_f1']:.4f}")
        
        print("\n📈 CRPS Statistics:")
        print(f"  Mean CRPS: {crps_metrics['mean_crps']:.4f}")
        print(f"  Std CRPS: {crps_metrics['std_crps']:.4f}")
        print(f"  CRPS Threshold: {crps_metrics['crps_threshold']:.4f}")
        print("="*60)
        
        # Combine metrics
        all_metrics = {**metrics, **crps_metrics}
        
        return all_metrics, y_pred_prob, crps_scores
    
    def calculate_crps_single(self, y_true, quantiles):
        """Calculate CRPS for a single prediction"""
        q10, q50, q90 = quantiles
        
        # CRPS approximation using quantiles
        crps = (q90 - q10) * 0.4 + abs(q50 - y_true)
        
        if y_true < q10:
            crps += (q10 - y_true) * 0.1
        elif y_true > q90:
            crps += (y_true - q90) * 0.1
            
        return crps
    
    def create_plots(self, history, metrics, y_test, y_pred_prob, crps_scores):
        """Create comprehensive visualization plots"""
        
        print("Creating CNN-CRPS visualization plots...")
        
        # 1. Training History Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎯 CNN-CRPS Model - Training Dashboard', fontsize=16, fontweight='bold')
        
        # Total Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
        axes[0, 0].set_title('📉 Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Attack Probability Accuracy
        axes[0, 1].plot(history.history['attack_prob_accuracy'], label='Training Accuracy', linewidth=2, color='#27ae60')
        axes[0, 1].plot(history.history['val_attack_prob_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
        axes[0, 1].set_title('📊 Classification Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Attack Probability Loss
        axes[0, 2].plot(history.history['attack_prob_loss'], label='Training', linewidth=2, color='#9b59b6')
        axes[0, 2].plot(history.history['val_attack_prob_loss'], label='Validation', linewidth=2, color='#e67e22')
        axes[0, 2].set_title('🎯 Classification Loss', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Quantile Losses
        quantiles = ['q10', 'q50', 'q90']
        colors = ['#1abc9c', '#34495e', '#e74c3c']
        
        for i, (q, color) in enumerate(zip(quantiles, colors)):
            axes[1, i].plot(history.history[f'{q}_loss'], label=f'Training {q.upper()}', linewidth=2, color=color)
            axes[1, i].plot(history.history[f'val_{q}_loss'], label=f'Validation {q.upper()}', linewidth=2, color=color, alpha=0.7)
            axes[1, i].set_title(f'📈 {q.upper()} Quantile Loss', fontweight='bold')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'cnn_crps_training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        plt.title('🎯 Confusion Matrix - CNN-CRPS Model\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Remove metrics text from confusion matrix
        
        plt.savefig(os.path.join(self.plots_save_path, 'cnn_crps_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'CNN-CRPS Model (AUC = {metrics["auc_roc"]:.3f})', linewidth=3, color='#e74c3c')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.7)
        plt.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('📈 ROC Curve - CNN-CRPS Model\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.plots_save_path, 'cnn_crps_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. CRPS Analysis
        plt.figure(figsize=(12, 8))
        plt.hist(crps_scores, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        plt.axvline(metrics['crps_threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'CRPS Threshold: {metrics["crps_threshold"]:.3f}')
        plt.xlabel('CRPS Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('📊 CRPS Score Distribution\nUncertainty Quantification Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        plt.text(0.7, 0.8, f'Mean CRPS: {metrics["mean_crps"]:.4f}\nStd CRPS: {metrics["std_crps"]:.4f}\nCRPS Accuracy: {metrics["crps_accuracy"]:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.savefig(os.path.join(self.plots_save_path, 'cnn_crps_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. CRPS-ES Time Series Plot with Global Threshold
        plt.figure(figsize=(12, 8))
        
        # Create time series of CRPS scores
        time_windows = np.arange(len(crps_scores))
        global_threshold = np.percentile(crps_scores, 95)  # 95th percentile as global threshold
        
        plt.plot(time_windows, crps_scores, 'b-', linewidth=1.5, label='CRPS-ES', alpha=0.8)
        plt.axhline(y=global_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Global Threshold = {global_threshold:.2f}')
        
        # Highlight anomalies (values above threshold)
        anomaly_indices = crps_scores > global_threshold
        if np.any(anomaly_indices):
            plt.scatter(time_windows[anomaly_indices], crps_scores[anomaly_indices], 
                       color='red', s=30, alpha=0.7, zorder=5)
        
        plt.xlabel('Time Window', fontsize=12)
        plt.ylabel('CRPS-ES Value', fontsize=12)
        plt.title('Client 1 - CRPS-ES with Global Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_save_path, 'crps_es_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Save all metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 
                      'CRPS_Accuracy', 'CRPS_Precision', 'CRPS_Recall', 'CRPS_F1_Score', 
                      'Mean_CRPS', 'Std_CRPS', 'CRPS_Threshold', 'Global_Threshold'],
            'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['auc_roc'], metrics['crps_accuracy'],
                     metrics['crps_precision'], metrics['crps_recall'], metrics['crps_f1'],
                     metrics['mean_crps'], metrics['std_crps'], metrics['crps_threshold'],
                     global_threshold]
        })
        
        metrics_df.to_csv(os.path.join(self.plots_save_path, 'model_evaluation_metrics.csv'), index=False)
        
        print("✅ CNN-CRPS plots created successfully!")
        print("✅ CRPS-ES time series plot generated!")
        print("✅ All metrics saved to CSV file!")

def main():
    """Main training function"""
    
    print("="*60)
    print("CNN-CRPS MODEL TRAINING")
    print("="*60)
    
    # Initialize model
    model_trainer = CNNCRPSModel()
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train_dict, y_val_dict, y_test = model_trainer.load_data()
    
    print("✅ Data loaded from output_data folder!")
        
    # Build model
    model_trainer.input_shape = (X_train.shape[1], X_train.shape[2])
    model_trainer.build_model()
        
    # Train model
    history = model_trainer.train_model(X_train, X_val, y_train_dict, y_val_dict)
        
    # Evaluate model
    metrics, y_pred_prob, crps_scores = model_trainer.evaluate_model(X_test, y_test)
        
    # Create plots
    model_trainer.create_plots(history, metrics, y_test, y_pred_prob, crps_scores)
        
    
    # Save model
    model_trainer.model.save(os.path.join(model_trainer.model_save_path, 'cnn_crps_model.h5'))
    
    print("\n" + "="*60)
    print("🎉 CNN-CRPS TRAINING COMPLETED!")
    print("="*60)
    print("✅ CNN model with CRPS uncertainty quantification trained")
    print("✅ Comprehensive evaluation with CRPS metrics completed")
    print("✅ Beautiful visualizations generated")
    print("✅ Model saved for deployment")
    
    if metrics['accuracy'] >= 0.8:
        print(f"🎯 TARGET ACHIEVED: {metrics['accuracy']:.1%} accuracy!")
    else:
        print(f"⚠️  Current accuracy: {metrics['accuracy']:.1%}")
    
    print(f"📊 CRPS-Enhanced Accuracy: {metrics['crps_accuracy']:.1%}")

if __name__ == "__main__":
    main()
