#!/usr/bin/env python3
"""
Test script to verify CNN-TCN model architecture
"""

import numpy as np
import tensorflow as tf
from cnn_tcn_model import create_model, CRPSMetrics

def test_model_creation():
    """Test model creation and basic functionality"""
    print("Testing CNN-TCN Model Architecture")
    print("="*40)
    
    # Test with sample input shape
    input_shape = (100, 50)  # 100 timesteps, 50 features
    
    try:
        # Create model
        model, model_builder = create_model(input_shape)
        
        print("✅ Model created successfully!")
        print(f"Input shape: {input_shape}")
        
        # Test with dummy data
        batch_size = 32
        dummy_input = np.random.randn(batch_size, *input_shape)
        
        # Forward pass
        predictions = model.predict(dummy_input, verbose=0)
        
        print("\nOutput shapes:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")
        
        # Test CRPS calculation
        crps_metrics = CRPSMetrics()
        
        # Dummy observations and predictions
        observations = np.random.randn(batch_size)
        q10_pred = predictions['q10'].flatten()
        q50_pred = predictions['q50'].flatten()
        q90_pred = predictions['q90'].flatten()
        
        crps_scores = crps_metrics.calculate_crps_quantiles(
            observations, q10_pred, q50_pred, q90_pred
        )
        
        print(f"\nCRPS scores shape: {crps_scores.shape}")
        print(f"Mean CRPS: {np.mean(crps_scores):.4f}")
        
        # Test global threshold computation
        threshold = crps_metrics.compute_global_threshold(crps_scores)
        print(f"Global threshold (95th percentile): {threshold:.4f}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_creation()
    
    if success:
        print("\n🎉 Model architecture is working correctly!")
        print("You can now proceed with data preprocessing and training.")
    else:
        print("\n💥 There are issues with the model. Please check the implementation.")
