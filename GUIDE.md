# DDoS Attack Detection in Fog Layer - Project Guide

## Project Overview
Enhanced CNN-TCN model with CRPS integration for DDoS attack detection using NSL-KDD dataset.

## Project Architecture

```
btp/
├── data/                           # Dataset storage
│   ├── KDDTrain+.txt              # NSL-KDD training data (raw)
│   ├── KDDTest+.txt               # NSL-KDD test data (raw)
│   ├── KDDTrain+_20Percent.txt    # NSL-KDD 20% training subset
│   ├── nsl_kdd_train.csv          # Converted training CSV
│   ├── nsl_kdd_test.csv           # Converted test CSV
│   ├── nsl_kdd_train_20percent.csv # Converted 20% subset CSV
│   └── dataset_info.json          # Dataset metadata
│
├── processed_data/                 # Preprocessed data for training
│   ├── X_train.npy                # Training features (1985, 100, 41)
│   ├── X_val.npy                  # Validation features (425, 100, 41)
│   ├── X_test.npy                 # Test features (426, 100, 41)
│   ├── y_train.npy                # Training labels
│   ├── y_val.npy                  # Validation labels
│   ├── y_test.npy                 # Test labels
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoders.pkl         # Categorical encoders
│   └── metadata.json              # Preprocessing metadata
│
├── models/                         # Trained models and metrics
│   ├── cnn_tcn_enhanced_model.h5  # Complete trained model
│   ├── model_weights.weights.h5   # Model weights only
│   ├── model_architecture.json    # Model architecture
│   ├── enhanced_model.tflite      # TensorFlow Lite model
│   ├── best_model.h5              # Best checkpoint during training
│   └── model_metrics.csv          # Training metrics
│
├── plots/                          # Generated visualizations
│   ├── training_dashboard.png     # Training metrics dashboard
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── roc_curve.png              # ROC curve analysis
│   ├── crps_analysis.png          # CRPS analysis dashboard
│   └── performance_comparison.png # Standard vs CRPS comparison
│
├── Core Files (New/Modified):
│   ├── download_nsl_kdd_dataset.py    # NSL-KDD dataset downloader
│   ├── preprocess_nsl_kdd_data.py     # NSL-KDD preprocessing pipeline
│   ├── train_enhanced_model.py        # Enhanced training script
│   └── cnn_tcn_model.py              # Enhanced CNN-TCN model (modified)
│
├── Legacy Files (Original):
│   ├── download_dataset.py           # Original BoT-IoT downloader
│   ├── preprocess_data.py            # Original preprocessing
│   ├── train_model.py                # Original training script
│   └── crps_evaluation.py            # Original CRPS evaluation
│
└── Configuration:
    ├── requirements.txt              # Python dependencies
    ├── setup_env.sh                 # Environment setup
    └── .gitignore                   # Git ignore rules
```

## Key New Files Created

### 1. Dataset Management
- **`download_nsl_kdd_dataset.py`**: Downloads NSL-KDD dataset without Kaggle authentication
- **`preprocess_nsl_kdd_data.py`**: Comprehensive preprocessing for NSL-KDD dataset

### 2. Enhanced Model Training
- **`train_enhanced_model.py`**: Enhanced training script with beautiful visualizations
- **`cnn_tcn_model.py`** (modified): Multi-output CNN-TCN model with CRPS integration

### 3. Generated Outputs
- **`data/`**: Contains NSL-KDD dataset (141,880 balanced samples)
- **`processed_data/`**: Preprocessed sequences (2,836 sequences of 100 timesteps × 41 features)
- **`models/`**: Trained models and metrics
- **`plots/`**: Beautiful visualization dashboards

## Dataset Location

**Primary Dataset Storage**: `/home/tushar/akshita/btp/data/`

### Raw Dataset Files:
- `KDDTrain+.txt` - NSL-KDD training data
- `KDDTest+.txt` - NSL-KDD test data
- `KDDTrain+_20Percent.txt` - 20% subset for quick testing

### Processed CSV Files:
- `nsl_kdd_train.csv` - Training data with proper headers
- `nsl_kdd_test.csv` - Test data with proper headers
- `nsl_kdd_train_20percent.csv` - 20% subset with headers

### Preprocessed Data: `/home/tushar/akshita/btp/processed_data/`
- Training: 1,985 sequences
- Validation: 425 sequences  
- Test: 426 sequences
- Features: 41 network flow features
- Sequence Length: 100 timesteps

## Model Architecture

### Enhanced CNN-TCN Features:
- **Input**: (100, 41) - 100 timesteps × 41 features
- **CNN Layers**: Feature extraction with batch normalization
- **TCN Layers**: Temporal dependency modeling
- **Multi-Output**: 4 outputs (attack_prob, q10, q50, q90)
- **CRPS Integration**: Uncertainty quantification
- **Parameters**: 42,980 (lightweight for edge deployment)

## Training Configuration

- **Dataset**: NSL-KDD (141,880 balanced samples)
- **Max Epochs**: 30 (with early stopping)
- **Batch Size**: 32
- **Regularization**: Dropout, L2, Batch Normalization
- **Callbacks**: Early stopping, LR reduction, checkpointing

## Usage Commands

1. **Download Dataset**:
   ```bash
   python download_nsl_kdd_dataset.py
   ```

2. **Preprocess Data**:
   ```bash
   python preprocess_nsl_kdd_data.py
   ```

3. **Train Model**:
   ```bash
   python train_enhanced_model.py
   ```

## Key Improvements

1. **Larger Dataset**: NSL-KDD (141K samples) vs BoT-IoT (small)
2. **Balanced Data**: Equal normal/attack samples (70,940 each)
3. **Enhanced Architecture**: Multi-output CNN-TCN with CRPS
4. **Overfitting Prevention**: Proper regularization and validation
5. **Beautiful Visualizations**: Comprehensive training dashboards
6. **Edge Deployment**: TensorFlow Lite model generation

## Expected Performance

- **Standard Accuracy**: ~80% (realistic, non-overfitted)
- **CRPS-Enhanced Metrics**: Uncertainty quantification
- **Model Size**: 167.89 KB (edge-friendly)
- **Training Time**: ~30 epochs with early stopping
