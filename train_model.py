#!/usr/bin/env python3
"""
Training script for CNN-TCN model with CRPS integration
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from cnn_tcn_model import create_model, CRPSMetrics
import properscoring as ps
from tqdm import tqdm
import joblib

class ModelTrainer:
    """CNN-TCN Model Trainer with CRPS integration"""
    
    def __init__(self, model_save_path="models", plots_save_path="plots"):
        self.model_save_path = model_save_path
        self.plots_save_path = plots_save_path
        self.model = None
        self.model_builder = None
        self.history = None
        self.crps_metrics = CRPSMetrics()
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(plots_save_path, exist_ok=True)
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        try:
            X_train = np.load("processed_data/X_train.npy")
            X_test = np.load("processed_data/X_test.npy")
            y_train = np.load("processed_data/y_train.npy")
            y_test = np.load("processed_data/y_test.npy")
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Test labels shape: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError:
            print("❌ Preprocessed data not found. Please run 'python preprocess_data.py' first.")
            return None, None, None, None
    
    def create_callbacks(self):
        """Create training callbacks including early stopping"""
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def compile_model(self, learning_rate=0.001):
        """Compile the CNN-TCN model"""
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'attack_prob': 'binary_crossentropy',
                'q10': 'mean_squared_error',
                'q50': 'mean_squared_error',
                'q90': 'mean_squared_error'
            },
            metrics={
                'attack_prob': ['accuracy', 'precision', 'recall']
            }
        )
        
        print("✅ Model compiled!")
        
    def train_model(self, X_train, X_test, y_train, y_test, epochs=30, batch_size=16):
        """Train the CNN-TCN model with improved settings"""
        
        print("Creating CNN-TCN model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model, self.model_builder = create_model(input_shape)
        
        # Compile the model
        self.compile_model()
        
        # Prepare training data for multi-output model
        train_targets = {
            'attack_prob': y_train,
            'q10': y_train.astype(np.float32),
            'q50': y_train.astype(np.float32),
            'q90': y_train.astype(np.float32)
        }
        
        val_targets = {
            'attack_prob': y_test,
            'q10': y_test.astype(np.float32),
            'q50': y_test.astype(np.float32),
            'q90': y_test.astype(np.float32)
        }
        
        print("Starting training...")
        callbacks = self.create_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, train_targets,
            validation_data=(X_test, val_targets),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with CRPS metrics"""
        
        print("Evaluating model performance...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Extract predictions
        y_pred_prob = predictions['attack_prob'].flatten()
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        q10_pred = predictions['q10'].flatten()
        q50_pred = predictions['q50'].flatten()
        q90_pred = predictions['q90'].flatten()
        
        # Calculate CRPS scores
        crps_scores = self.crps_metrics.calculate_crps_quantiles(
            y_test, q10_pred, q50_pred, q90_pred
        )
        
        # Compute global CRPS threshold
        global_threshold = self.crps_metrics.compute_global_threshold(crps_scores)
        
        # CRPS-based anomaly detection
        crps_anomalies = crps_scores > global_threshold
        
        # Combined decision: (p_attack > 0.5) AND (CRPS > threshold)
        combined_predictions = (y_pred_prob > 0.5) & crps_anomalies
        
        # Calculate metrics
        metrics = {}
        
        # Standard classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
        metrics['precision'] = precision_score(y_test, y_pred_binary)
        metrics['recall'] = recall_score(y_test, y_pred_binary)
        metrics['f1_score'] = f1_score(y_test, y_pred_binary)
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_prob)
        
        # CRPS-enhanced metrics
        metrics['crps_accuracy'] = accuracy_score(y_test, combined_predictions)
        metrics['crps_precision'] = precision_score(y_test, combined_predictions)
        metrics['crps_recall'] = recall_score(y_test, combined_predictions)
        metrics['crps_f1_score'] = f1_score(y_test, combined_predictions)
        
        # CRPS statistics
        metrics['mean_crps'] = np.mean(crps_scores)
        metrics['std_crps'] = np.std(crps_scores)
        metrics['global_threshold'] = global_threshold
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\nStandard Classification Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print("\nCRPS-Enhanced Metrics:")
        print(f"CRPS Accuracy: {metrics['crps_accuracy']:.4f}")
        print(f"CRPS Precision: {metrics['crps_precision']:.4f}")
        print(f"CRPS Recall: {metrics['crps_recall']:.4f}")
        print(f"CRPS F1-Score: {metrics['crps_f1_score']:.4f}")
        
        print(f"\nCRPS Statistics:")
        print(f"Mean CRPS: {metrics['mean_crps']:.4f}")
        print(f"Std CRPS: {metrics['std_crps']:.4f}")
        print(f"Global Threshold: {metrics['global_threshold']:.4f}")
        
        return metrics, predictions, crps_scores
    
    def save_metrics(self, metrics):
        """Save metrics to CSV file"""
        
        metrics_df = pd.DataFrame([metrics])
        metrics_file = os.path.join(self.model_save_path, 'model_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"✅ Metrics saved to {metrics_file}")
        
    def create_plots(self, X_test, y_test, predictions, crps_scores, metrics):
        """Create and save model evaluation plots"""
        
        print("Creating evaluation plots...")
        
        # 1. Training history plots
        if self.history is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Accuracy plot
            axes[0, 1].plot(self.history.history['attack_prob_accuracy'], label='Training Accuracy')
            axes[0, 1].plot(self.history.history['val_attack_prob_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            
            # Precision plot
            axes[1, 0].plot(self.history.history['attack_prob_precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_attack_prob_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            
            # Recall plot
            axes[1, 1].plot(self.history.history['attack_prob_recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_attack_prob_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confusion Matrix
        y_pred_prob = predictions['attack_prob'].flatten()
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.plots_save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. CRPS Distribution
        plt.figure(figsize=(12, 8))
        
        # CRPS histogram
        plt.subplot(2, 2, 1)
        plt.hist(crps_scores, bins=50, alpha=0.7, color='blue')
        plt.axvline(metrics['global_threshold'], color='red', linestyle='--', 
                   label=f'Global Threshold = {metrics["global_threshold"]:.3f}')
        plt.xlabel('CRPS Score')
        plt.ylabel('Frequency')
        plt.title('CRPS Score Distribution')
        plt.legend()
        
        # CRPS vs True Labels
        plt.subplot(2, 2, 2)
        normal_crps = crps_scores[y_test == 0]
        attack_crps = crps_scores[y_test == 1]
        
        plt.boxplot([normal_crps, attack_crps], labels=['Normal', 'Attack'])
        plt.ylabel('CRPS Score')
        plt.title('CRPS Scores by True Label')
        
        # CRPS Time Series (sample)
        plt.subplot(2, 2, 3)
        sample_size = min(1000, len(crps_scores))
        sample_indices = np.random.choice(len(crps_scores), sample_size, replace=False)
        sample_crps = crps_scores[sample_indices]
        sample_labels = y_test[sample_indices]
        
        plt.plot(sample_crps, alpha=0.7)
        plt.axhline(metrics['global_threshold'], color='red', linestyle='--', 
                   label=f'Threshold = {metrics["global_threshold"]:.3f}')
        plt.xlabel('Sample Index')
        plt.ylabel('CRPS Score')
        plt.title('CRPS Scores Over Time (Sample)')
        plt.legend()
        
        # Quantile predictions
        plt.subplot(2, 2, 4)
        q10_pred = predictions['q10'].flatten()
        q50_pred = predictions['q50'].flatten()
        q90_pred = predictions['q90'].flatten()
        
        sample_q10 = q10_pred[sample_indices]
        sample_q50 = q50_pred[sample_indices]
        sample_q90 = q90_pred[sample_indices]
        
        plt.plot(sample_q10, label='Q10', alpha=0.7)
        plt.plot(sample_q50, label='Q50', alpha=0.7)
        plt.plot(sample_q90, label='Q90', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Quantile Prediction')
        plt.title('Quantile Predictions (Sample)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'crps_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Plots saved to plots/ directory")
    
    def save_model(self):
        """Save the trained model"""
        
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return
        
        # Save full model
        model_path = os.path.join(self.model_save_path, 'cnn_tcn_model.h5')
        self.model.save(model_path)
        
        # Save model weights
        weights_path = os.path.join(self.model_save_path, 'model_weights.weights.h5')
        self.model.save_weights(weights_path)
        
        # Save model architecture
        architecture_path = os.path.join(self.model_save_path, 'model_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(self.model.to_json())
        
        print(f"✅ Model saved:")
        print(f"  - Full model: {model_path}")
        print(f"  - Weights: {weights_path}")
        print(f"  - Architecture: {architecture_path}")
        
        # Convert to TensorFlow Lite for edge deployment
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            tflite_path = os.path.join(self.model_save_path, 'model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"  - TensorFlow Lite: {tflite_path}")
            
        except Exception as e:
            print(f"⚠️  TensorFlow Lite conversion failed: {e}")

def main():
    """Main training function"""
    
    print("CNN-TCN Model Training with CRPS Integration")
    print("="*50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    if X_train is None:
        return
    
    # Train model
    history = trainer.train_model(X_train, X_test, y_train, y_test, epochs=30)
    
    # Evaluate model
    metrics, predictions, crps_scores = trainer.evaluate_model(X_test, y_test)
    
    # Save metrics
    trainer.save_metrics(metrics)
    
    # Create plots
    trainer.create_plots(X_test, y_test, predictions, crps_scores, metrics)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("✅ Model trained and saved successfully!")
    print("✅ Metrics saved to CSV")
    print("✅ Plots generated and saved")
    print("✅ Model ready for federated learning integration")
    
    print(f"\nModel Performance Summary:")
    print(f"  - Standard Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - CRPS-Enhanced Accuracy: {metrics['crps_accuracy']:.4f}")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  - Global CRPS Threshold: {metrics['global_threshold']:.4f}")

if __name__ == "__main__":
    main()
