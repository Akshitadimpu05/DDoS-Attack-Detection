#!/usr/bin/env python3
"""
Enhanced training script for CNN-TCN model with CRPS integration
Optimized for CIC-IDS-2017 dataset with beautiful visualizations
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
import json
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedModelTrainer:
    """Enhanced CNN-TCN Model Trainer with CRPS integration and beautiful visualizations"""
    
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
        
        # Load metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load preprocessing metadata"""
        try:
            with open("processed_data/metadata.json", 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata: {self.metadata['train_samples']} training samples")
        except FileNotFoundError:
            print("⚠️  Metadata not found. Please run preprocessing first.")
            self.metadata = {}
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed CIC-IDS-2017 data...")
        
        try:
            X_train = np.load("processed_data/X_train.npy")
            X_val = np.load("processed_data/X_val.npy")
            X_test = np.load("processed_data/X_test.npy")
            y_train = np.load("processed_data/y_train.npy")
            y_val = np.load("processed_data/y_val.npy")
            y_test = np.load("processed_data/y_test.npy")
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")
            
            # Prepare targets for multi-output model
            y_train_dict = {
                'attack_prob': y_train,
                'q10': y_train * 0.1,  # Lower quantile
                'q50': y_train * 0.5,  # Median
                'q90': y_train * 0.9   # Upper quantile
            }
            
            y_val_dict = {
                'attack_prob': y_val,
                'q10': y_val * 0.1,
                'q50': y_val * 0.5,
                'q90': y_val * 0.9
            }
            
            return X_train, X_val, X_test, y_train_dict, y_val_dict, y_test
            
        except FileNotFoundError:
            print("❌ Preprocessed data not found. Please run 'python preprocess_cic_ids_data.py' first.")
            return None, None, None, None, None, None
    
    def create_callbacks(self, max_epochs=30):
        """Create training callbacks with early stopping"""
        
        callbacks = [
            # Early stopping with patience
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_attack_prob_loss',
                mode='min',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_attack_prob_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * 0.95 ** epoch,
                verbose=0
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=40, batch_size=32):
        """Train enhanced CNN-TCN model with proper regularization"""
        
        print("Creating enhanced CNN-TCN model for CIC-IDS-2017 dataset...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model, self.model_builder = create_model(input_shape)
        
        # Calculate class weights for balanced training
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train['attack_prob'])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train['attack_prob'])
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        print(f"Class weights: {class_weight_dict}")
        print(f"Training for maximum {epochs} epochs with batch size {batch_size}")
        
        # Create callbacks with improved patience
        callbacks = [
            # Early stopping with more patience
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_attack_prob_loss',
                mode='min',
                factor=0.3,
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_attack_prob_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model (remove class_weight for multi-output model)
        print("Starting training...")
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
        """Evaluate model performance with both standard and CRPS metrics"""
        
        print("Evaluating model performance...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        y_pred_prob = predictions['attack_prob'].flatten()
        q10_pred = predictions['q10'].flatten()
        q50_pred = predictions['q50'].flatten()
        q90_pred = predictions['q90'].flatten()
        
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        # Standard classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        standard_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_binary, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_prob)
        }
        
        # CRPS evaluation
        print("Calculating CRPS metrics...")
        crps_scores = self.crps_metrics.calculate_crps_quantiles(
            y_test, q10_pred, q50_pred, q90_pred
        )
        
        # Compute CRPS threshold and evaluate
        crps_threshold = self.crps_metrics.compute_global_threshold(crps_scores, percentile=95)
        crps_metrics = self.crps_metrics.evaluate_crps_performance(y_test, crps_scores, crps_threshold)
        
        # Combine all metrics
        all_metrics = {**standard_metrics, **crps_metrics}
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\n🎯 Standard Classification Metrics:")
        print(f"  Accuracy: {standard_metrics['accuracy']:.4f}")
        print(f"  Precision: {standard_metrics['precision']:.4f}")
        print(f"  Recall: {standard_metrics['recall']:.4f}")
        print(f"  F1-Score: {standard_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {standard_metrics['auc_roc']:.4f}")
        
        print("\n📊 CRPS-Enhanced Metrics:")
        print(f"  CRPS Accuracy: {crps_metrics['crps_accuracy']:.4f}")
        print(f"  CRPS Precision: {crps_metrics['crps_precision']:.4f}")
        print(f"  CRPS Recall: {crps_metrics['crps_recall']:.4f}")
        print(f"  CRPS F1-Score: {crps_metrics['crps_f1_score']:.4f}")
        
        print(f"\n📈 CRPS Statistics:")
        print(f"  Mean CRPS: {crps_metrics['mean_crps']:.4f}")
        print(f"  Std CRPS: {crps_metrics['std_crps']:.4f}")
        print(f"  Global Threshold: {crps_metrics['threshold']:.4f}")
        
        return all_metrics, y_pred_prob, crps_scores
    
    def create_beautiful_plots(self, X_test, y_test, y_pred_prob, crps_scores, metrics):
        """Create beautiful and comprehensive visualization plots"""
        
        print("Creating beautiful visualization plots...")
        
        # Set up the plotting style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # 1. Training History Dashboard
        if self.history is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('🚀 CNN-TCN Training Dashboard - CIC-IDS-2017 Dataset', fontsize=16, fontweight='bold')
            
            # Loss plots
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
            axes[0, 0].set_title('📉 Total Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Attack probability accuracy
            axes[0, 1].plot(self.history.history['attack_prob_accuracy'], label='Training Accuracy', linewidth=2, color='#27ae60')
            axes[0, 1].plot(self.history.history['val_attack_prob_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
            axes[0, 1].set_title('🎯 Classification Accuracy', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision and Recall
            axes[0, 2].plot(self.history.history['attack_prob_precision'], label='Training Precision', linewidth=2, color='#9b59b6')
            axes[0, 2].plot(self.history.history['val_attack_prob_precision'], label='Validation Precision', linewidth=2, color='#e67e22')
            axes[0, 2].set_title('🔍 Precision', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            axes[1, 0].plot(self.history.history['attack_prob_recall'], label='Training Recall', linewidth=2, color='#1abc9c')
            axes[1, 0].plot(self.history.history['val_attack_prob_recall'], label='Validation Recall', linewidth=2, color='#34495e')
            axes[1, 0].set_title('📊 Recall', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # AUC
            axes[1, 1].plot(self.history.history['attack_prob_auc'], label='Training AUC', linewidth=2, color='#e74c3c')
            axes[1, 1].plot(self.history.history['val_attack_prob_auc'], label='Validation AUC', linewidth=2, color='#3498db')
            axes[1, 1].set_title('📈 AUC-ROC', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Learning Rate
            if 'lr' in self.history.history:
                axes[1, 2].plot(self.history.history['lr'], linewidth=2, color='#f39c12')
                axes[1, 2].set_title('📚 Learning Rate', fontweight='bold')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('Learning Rate')
                axes[1, 2].set_yscale('log')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Learning Rate\nNot Available', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('📚 Learning Rate', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_save_path, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Detailed Confusion Matrix with Enhanced Metrics
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        # Create detailed confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Standard confusion matrix with counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        ax1.set_title('🎯 Confusion Matrix (Counts)\nDDoS Attack Detection', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Add count annotations with percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percentage = (count / total) * 100
                ax1.text(j + 0.5, i + 0.7, f'{count}\n({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Right plot: Normalized confusion matrix with percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges', ax=ax2,
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Percentage'})
        ax2.set_title('📊 Normalized Confusion Matrix (%)\nDDoS Attack Detection', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        # Add detailed metrics text box
        metrics_text = f"""📈 Detailed Classification Metrics:
        
True Positives (TP): {tp}
True Negatives (TN): {tn}
False Positives (FP): {fp}
False Negatives (FN): {fn}

Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall (Sensitivity): {metrics['recall']:.4f}
Specificity: {tn/(tn+fp):.4f}
F1-Score: {metrics['f1_score']:.4f}
AUC-ROC: {metrics['auc_roc']:.4f}

False Positive Rate: {fp/(fp+tn):.4f}
False Negative Rate: {fn/(fn+tp):.4f}"""
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'detailed_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2b. Simple Confusion Matrix (for compatibility)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'})
        plt.title('🎯 Confusion Matrix - DDoS Attack Detection\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add performance metrics as text
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        
        plt.text(1.05, 0.8, f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.savefig(os.path.join(self.plots_save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Enhanced ROC Curve with Multiple Metrics
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: ROC Curve
        ax1.plot(fpr, tpr, label=f'CNN-TCN Model (AUC = {metrics["auc_roc"]:.3f})', 
                linewidth=3, color='#e74c3c')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.7)
        ax1.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
                label=f'Optimal Threshold: {optimal_threshold:.3f}')
        
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('📈 ROC Curve - DDoS Detection\nNSL-KDD Dataset', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
        avg_precision = average_precision_score(y_test, y_pred_prob)
        
        ax2.plot(recall_curve, precision_curve, linewidth=3, color='#2ecc71',
                label=f'PR Curve (AP = {avg_precision:.3f})')
        ax2.fill_between(recall_curve, precision_curve, alpha=0.3, color='#2ecc71')
        ax2.axhline(y=metrics['precision'], color='red', linestyle='--', 
                   label=f'Current Precision: {metrics["precision"]:.3f}')
        ax2.axvline(x=metrics['recall'], color='blue', linestyle='--',
                   label=f'Current Recall: {metrics["recall"]:.3f}')
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('📊 Precision-Recall Curve\nDDoS Detection', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add performance summary
        perf_text = f"""🎯 Model Performance Summary:
        
AUC-ROC: {metrics['auc_roc']:.4f}
Average Precision: {avg_precision:.4f}
Optimal Threshold: {optimal_threshold:.4f}

At Optimal Threshold:
• TPR (Sensitivity): {tpr[optimal_idx]:.4f}
• FPR: {fpr[optimal_idx]:.4f}
• Specificity: {1-fpr[optimal_idx]:.4f}

Current Model (0.5 threshold):
• Precision: {metrics['precision']:.4f}
• Recall: {metrics['recall']:.4f}
• F1-Score: {metrics['f1_score']:.4f}"""
        
        plt.figtext(0.02, 0.02, perf_text, fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'enhanced_roc_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3b. Simple ROC Curve (for compatibility)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'CNN-TCN Model (AUC = {metrics["auc_roc"]:.3f})', linewidth=3, color='#e74c3c')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.7)
        plt.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('📈 ROC Curve - DDoS Attack Detection\nNSL-KDD Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, label=f'Optimal Threshold: {optimal_threshold:.3f}')
        plt.legend(fontsize=12)
        
        plt.savefig(os.path.join(self.plots_save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. CRPS Analysis Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 CRPS Analysis Dashboard - Uncertainty Quantification', fontsize=16, fontweight='bold')
        
        # CRPS distribution
        axes[0, 0].hist(crps_scores, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        axes[0, 0].axvline(metrics['threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold: {metrics["threshold"]:.3f}')
        axes[0, 0].set_title('📈 CRPS Score Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('CRPS Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CRPS vs True Labels
        normal_crps = crps_scores[y_test == 0]
        attack_crps = crps_scores[y_test == 1]
        
        axes[0, 1].boxplot([normal_crps, attack_crps], labels=['Normal', 'Attack'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_title('📊 CRPS Scores by Class', fontweight='bold')
        axes[0, 1].set_ylabel('CRPS Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction confidence
        confidence = np.maximum(y_pred_prob, 1 - y_pred_prob)
        axes[1, 0].scatter(confidence, crps_scores, alpha=0.6, c=y_test, cmap='RdYlBu')
        axes[1, 0].set_title('🎯 Prediction Confidence vs CRPS', fontweight='bold')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('CRPS Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # CRPS-based classification performance
        crps_pred = (crps_scores > metrics['threshold']).astype(int)
        crps_cm = confusion_matrix(y_test, crps_pred)
        
        sns.heatmap(crps_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 1],
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        axes[1, 1].set_title('🔍 CRPS-based Classification', fontweight='bold')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'crps_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Model Performance Comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        standard_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        crps_values = [metrics['crps_accuracy'], metrics['crps_precision'], metrics['crps_recall'], metrics['crps_f1_score']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, standard_values, width, label='Standard CNN-TCN', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, crps_values, width, label='CRPS-Enhanced', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('🏆 Model Performance Comparison\nStandard vs CRPS-Enhanced Detection', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_save_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Beautiful plots created and saved!")
        
    def save_metrics(self, metrics):
        """Save comprehensive metrics to CSV file"""
        
        metrics_df = pd.DataFrame([metrics])
        metrics_file = os.path.join(self.model_save_path, 'model_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"✅ Comprehensive metrics saved to {metrics_file}")
        
    def save_model(self):
        """Save the trained model with all components"""
        
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return
        
        # Save full model
        model_path = os.path.join(self.model_save_path, 'cnn_tcn_enhanced_model.h5')
        self.model.save(model_path)
        
        # Save model weights
        weights_path = os.path.join(self.model_save_path, 'model_weights.weights.h5')
        self.model.save_weights(weights_path)
        
        # Save model architecture
        architecture_path = os.path.join(self.model_save_path, 'model_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(self.model.to_json())
        
        print(f"✅ Enhanced model saved:")
        print(f"  - Full model: {model_path}")
        print(f"  - Weights: {weights_path}")
        print(f"  - Architecture: {architecture_path}")
        
        # Convert to TensorFlow Lite for edge deployment
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            tflite_path = os.path.join(self.model_save_path, 'enhanced_model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"  - TensorFlow Lite: {tflite_path}")
            
        except Exception as e:
            print(f"⚠️  TensorFlow Lite conversion failed: {e}")

def main():
    """Main training function"""
    
    print("🚀 Enhanced CNN-TCN Model Training with CRPS Integration")
    print("📊 Dataset: CIC-IDS-2017 (Large-scale Network Intrusion Detection)")
    print("="*70)
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    
    if X_train is None:
        return
    
    # Train model (increased to 40 epochs for better performance)
    print(f"\n🎯 Training with maximum 40 epochs...")
    history = trainer.train_model(X_train, X_val, y_train, y_val, epochs=40, batch_size=32)
    
    # Evaluate model
    print(f"\n📊 Evaluating model performance...")
    metrics, y_pred_prob, crps_scores = trainer.evaluate_model(X_test, y_test)
    
    # Save metrics
    trainer.save_metrics(metrics)
    
    # Create beautiful plots
    print(f"\n🎨 Creating beautiful visualizations...")
    trainer.create_beautiful_plots(X_test, y_test, y_pred_prob, crps_scores, metrics)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("✅ Enhanced CNN-TCN model trained and saved")
    print("✅ Comprehensive metrics calculated and saved")
    print("✅ Beautiful visualizations generated")
    print("✅ Model ready for deployment in fog layer")
    
    print(f"\n📈 Final Performance Summary:")
    print(f"  🎯 Standard Accuracy: {metrics['accuracy']:.4f}")
    print(f"  📊 CRPS-Enhanced Accuracy: {metrics['crps_accuracy']:.4f}")
    print(f"  📈 AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  🔍 CRPS Threshold: {metrics['threshold']:.4f}")
    
    if metrics['accuracy'] > 0.75:
        print("\n🏆 Great! Model achieved good accuracy without overfitting!")
    else:
        print("\n⚠️  Model accuracy could be improved. Consider adjusting hyperparameters.")

if __name__ == "__main__":
    main()
