# IoT DDoS Detection with CNN-TCN and CRPS Integration

This project implements a lightweight CNN-TCN (Convolutional Neural Network - Temporal Convolutional Network) model for IoT DDoS detection with CRPS (Continuous Ranked Probability Score) integration, designed for federated learning deployment on Raspberry Pi devices.

## Project Overview

The system uses the Bot-IoT dataset to train a hybrid CNN-TCN model that:
- Captures local flow/packet patterns using CNN layers
- Models long-term dependencies with TCN layers using dilated causal convolutions
- Integrates CRPS metrics for confidence calibration and robust anomaly detection
- Maintains lightweight architecture (<100k parameters) for edge deployment

## Features

- **Hybrid CNN-TCN Architecture**: Combines spatial and temporal pattern recognition
- **CRPS Integration**: Provides confidence calibration to reduce false alarms
- **Multi-output Model**: Predicts attack probability and quantiles (q10, q50, q90)
- **Early Stopping**: Prevents overfitting with automatic training termination
- **Comprehensive Evaluation**: Includes standard and CRPS-enhanced metrics
- **Edge-Ready**: TensorFlow Lite conversion for Raspberry Pi deployment
- **Federated Learning Ready**: Model architecture suitable for FL integration

## Dataset

The project uses the Bot-IoT dataset from UNSW Canberra Cyber Range Lab:
- **Source**: [Kaggle - Bot-IoT Dataset](https://www.kaggle.com/datasets/vigneshvenkateswaran/bot-iot)
- **Focus**: DDoS/DoS attacks and normal traffic
- **Features**: Flow-based features (packet count, byte count, duration, flags, entropy, ratios)

## Installation and Setup

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd /path/to/btp

# Make setup script executable
chmod +x setup_env.sh

# Run setup script
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Kaggle API Setup

Before downloading the dataset, set up Kaggle API credentials:

```bash
# Go to https://www.kaggle.com/account
# Click 'Create New API Token'
# Download kaggle.json
# Place it in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Usage Instructions

### Step 1: Download Dataset

```bash
python download_dataset.py
```

This script will:
- Download the Bot-IoT dataset from Kaggle
- Extract and organize the data
- Provide dataset exploration and statistics

### Step 2: Preprocess Data

```bash
python preprocess_data.py
```

This script will:
- Filter relevant classes (DDoS/DoS and normal traffic)
- Extract flow-based features
- Create time windows for sequence modeling
- Normalize features using StandardScaler
- Split data into train/test sets
- Save processed data to `processed_data/` directory

### Step 3: Train Model

```bash
python train_model.py
```

This script will:
- Create CNN-TCN model with CRPS integration
- Train the model with early stopping
- Evaluate performance with standard and CRPS-enhanced metrics
- Generate comprehensive plots and visualizations
- Save model metrics to CSV
- Save trained model in multiple formats (H5, TFLite)

## Model Architecture

### CNN Layers
- **Conv1D(32, 3)** + BatchNorm + MaxPooling
- **Conv1D(64, 3)** + BatchNorm + MaxPooling
- Captures local patterns in network flows

### TCN Layers
- **Dilated Convolutions** (rates: 1, 2, 4)
- **Residual Connections** for gradient flow
- **Causal Padding** for temporal modeling
- Captures long-term dependencies

### Output Heads
- **Attack Probability**: Sigmoid activation for binary classification
- **Quantiles (q10, q50, q90)**: Linear activation for CRPS calculation

## CRPS Integration

The model integrates CRPS (Continuous Ranked Probability Score) for:
- **Confidence Calibration**: Reduces false alarms on noisy IoT traffic
- **Robust Decision Making**: Combined rule: (p_attack > 0.5) AND (CRPS > θ_global)
- **Global Thresholding**: Computed from quantile predictions

## Output Files

After training, the following files will be generated:

### Models (`models/` directory)
- `cnn_tcn_model.h5` - Full Keras model
- `model_weights.h5` - Model weights only
- `model_architecture.json` - Model architecture
- `model.tflite` - TensorFlow Lite model for edge deployment
- `model_metrics.csv` - Training and evaluation metrics

### Plots (`plots/` directory)
- `training_history.png` - Loss, accuracy, precision, recall curves
- `confusion_matrix.png` - Classification confusion matrix
- `roc_curve.png` - ROC curve with AUC score
- `crps_analysis.png` - CRPS distribution and analysis
- `label_distribution.png` - Dataset label distribution
- `feature_correlation.png` - Feature correlation heatmap

### Processed Data (`processed_data/` directory)
- `X_train.npy`, `X_test.npy` - Training and test features
- `y_train.npy`, `y_test.npy` - Training and test labels
- `scaler.pkl` - Fitted StandardScaler for feature normalization

## Model Performance Metrics

The system evaluates performance using:

### Standard Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC for binary classification

### CRPS-Enhanced Metrics
- CRPS-based accuracy, precision, recall
- Global CRPS threshold for anomaly detection
- Mean and standard deviation of CRPS scores

## Federated Learning Integration

The trained model is ready for federated learning with:
- Lightweight architecture (<100k parameters)
- TensorFlow Lite conversion for Raspberry Pi
- Model weight extraction for FL aggregation
- CRPS threshold computation for global coordination

## Troubleshooting

### Common Issues

1. **Kaggle API Error**
   - Ensure kaggle.json is properly placed and has correct permissions
   - Verify Kaggle account has API access enabled

2. **Memory Issues**
   - Reduce batch size in training script
   - Use smaller time windows in preprocessing

3. **TensorFlow Lite Conversion Error**
   - Model will still be saved in H5 format
   - TFLite conversion is optional for edge deployment

### Dependencies

If you encounter package conflicts, ensure you're using:
- Python 3.8+
- TensorFlow 2.13.0
- Compatible versions as specified in requirements.txt

## Next Steps

After successful training:
1. **Model Quantization**: Convert to INT8 for further optimization
2. **Federated Learning**: Integrate with FL framework (FedAdam/FedProx)
3. **Raspberry Pi Deployment**: Deploy TFLite model on edge devices
4. **Real-time Inference**: Implement streaming inference pipeline

## Project Structure

```
btp/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_env.sh             # Environment setup script
├── download_dataset.py       # Dataset downloader
├── preprocess_data.py        # Data preprocessing
├── cnn_tcn_model.py         # Model architecture
├── train_model.py           # Training script
├── data/                    # Downloaded dataset
├── processed_data/          # Preprocessed data
├── models/                  # Trained models
└── plots/                   # Generated plots
```

## References

- Bot-IoT Dataset: UNSW Canberra Cyber Range Lab
- CRPS: Continuous Ranked Probability Score for probabilistic forecasting
- TCN: Temporal Convolutional Networks for sequence modeling
- Federated Learning: Distributed machine learning framework
