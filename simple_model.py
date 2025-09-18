#!/usr/bin/env python3
"""
Simplified high-performance model for DDoS detection
Designed for 80%+ accuracy on NSL-KDD dataset
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

class SimpleDDoSModel:
    """Simplified high-performance DDoS detection model"""
    
    def __init__(self, input_shape=(40,), model_save_path="models", plots_save_path="plots"):
        self.input_shape = input_shape
        self.model_save_path = model_save_path
        self.plots_save_path = plots_save_path
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(plots_save_path, exist_ok=True)
    
    def build_model(self):
        """Build optimized neural network for high accuracy"""
        
        print("Building optimized DDoS detection model...")
        
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Dense layers with proper regularization
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='attack_prob')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with optimized settings
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model built successfully")
        print(f"Model parameters: {self.model.count_params():,}")
        
        return self.model
    
    def load_data(self):
        """Load processed data"""
        
        print("Loading processed data...")
        
        try:
            X_train = np.load(os.path.join('processed_data', 'X_train.npy'))
            X_val = np.load(os.path.join('processed_data', 'X_val.npy'))
            X_test = np.load(os.path.join('processed_data', 'X_test.npy'))
            y_train = np.load(os.path.join('processed_data', 'y_train.npy'))
            y_val = np.load(os.path.join('processed_data', 'y_val.npy'))
            y_test = np.load(os.path.join('processed_data', 'y_test.npy'))
            
            # Flatten sequences to simple feature vectors
            if len(X_train.shape) == 3:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_val = X_val.reshape(X_val.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
            
            print(f"✅ Data loaded:")
            print(f"  Training: {X_train.shape}")
            print(f"  Validation: {X_val.shape}")
            print(f"  Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None, None, None, None
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=100, batch_size=64):
        """Train the model with optimized settings"""
        
        print(f"Training model for {epochs} epochs...")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=15, 
                restore_best_weights=True, 
                verbose=1,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=8, 
                min_lr=1e-7, 
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_simple_model.h5'),
                monitor='val_accuracy',
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
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        
        print("Evaluating model...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_prob)
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📊 Precision: {metrics['precision']:.4f}")
        print(f"🔍 Recall: {metrics['recall']:.4f}")
        print(f"⚖️  F1-Score: {metrics['f1_score']:.4f}")
        print(f"📈 AUC-ROC: {metrics['auc_roc']:.4f}")
        print("="*50)
        
        return metrics, y_pred_prob.flatten()
    
    def create_plots(self, history, metrics, y_test, y_pred_prob):
        """Create comprehensive visualization plots"""
        
        print("Creating visualization plots...")
        
        # 1. Training History
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Simple DDoS Model - Training Dashboard', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
        axes[0, 0].set_title('📉 Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#27ae60')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
        axes[0, 1].set_title('📊 Model Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2, color='#9b59b6')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2, color='#e67e22')
        axes[1, 0].set_title('🎯 Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2, color='#1abc9c')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2, color='#34495e')
        axes[1, 1].set_title('🔍 Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'simple_training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        plt.title('🎯 Confusion Matrix - Simple DDoS Model\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add metrics text
        plt.text(1.05, 0.8, f'Accuracy: {metrics["accuracy"]:.3f}\nPrecision: {metrics["precision"]:.3f}\nRecall: {metrics["recall"]:.3f}\nF1-Score: {metrics["f1_score"]:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.savefig(os.path.join(self.plots_save_path, 'simple_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'Simple Model (AUC = {metrics["auc_roc"]:.3f})', linewidth=3, color='#e74c3c')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.7)
        plt.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('📈 ROC Curve - Simple DDoS Model\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.plots_save_path, 'simple_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Plots created successfully!")

def main():
    """Main training function"""
    
    print("="*60)
    print("SIMPLE HIGH-PERFORMANCE DDOS MODEL TRAINING")
    print("="*60)
    
    # Initialize model
    model_trainer = SimpleDDoSModel()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = model_trainer.load_data()
    
    if X_train is None:
        print("❌ Failed to load data. Please run preprocessing first.")
        return
    
    # Update input shape based on actual data
    model_trainer.input_shape = (X_train.shape[1],)
    
    # Build model
    model_trainer.build_model()
    
    # Train model
    history = model_trainer.train_model(X_train, X_val, y_train, y_val, epochs=100, batch_size=64)
    
    # Evaluate model
    metrics, y_pred_prob = model_trainer.evaluate_model(X_test, y_test)
    
    # Create plots
    model_trainer.create_plots(history, metrics, y_test, y_pred_prob)
    
    # Save model
    model_trainer.model.save(os.path.join(model_trainer.model_save_path, 'simple_ddos_model.h5'))
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Simple high-performance model trained")
    print("✅ Comprehensive evaluation completed")
    print("✅ Beautiful visualizations generated")
    print("✅ Model saved for deployment")
    
    if metrics['accuracy'] >= 0.8:
        print(f"🎯 TARGET ACHIEVED: {metrics['accuracy']:.1%} accuracy!")
    else:
        print(f"⚠️  Current accuracy: {metrics['accuracy']:.1%} - may need further tuning")

if __name__ == "__main__":
    main()
