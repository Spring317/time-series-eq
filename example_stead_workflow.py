#!/usr/bin/env python3
"""
Quick example demonstrating STEAD+DAS training workflow
"""

def example_workflow():
    """
    This example shows the complete workflow for training on STEAD
    and testing on DAS in a simplified manner
    """
    
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║          STEAD → DAS Transfer Learning Workflow                       ║
╚═══════════════════════════════════════════════════════════════════════╝

This example demonstrates how to:
1. Load STEAD dataset (large, diverse earthquake data)
2. Train an LSTM model on STEAD
3. Test the trained model on your local DAS dataset

═══════════════════════════════════════════════════════════════════════

STEP 1: Setup
──────────────

First, ensure STEAD dataset is downloaded:
  
  $ python setup_stead.py
  
  # or
  
  $ bash setup_stead.sh

This downloads ~85GB of earthquake data with 1.2M waveforms.

═══════════════════════════════════════════════════════════════════════

STEP 2: Verify Your DAS Data
─────────────────────────────

Ensure you have:
  ✓ DAS files in data/ directory
  ✓ labels.json with classifications
  
  $ python generate_labels.py  # if labels.json doesn't exist

═══════════════════════════════════════════════════════════════════════

STEP 3: Run Training
────────────────────

Train on STEAD, test on DAS:
  
  $ python train_stead_test_das.py

This will:
  • Load STEAD (3-channel data, 6000 samples)
  • Adapt to DAS channel count (auto-detected)
  • Train LSTM model (typically 30-50 epochs)
  • Test on DAS dataset
  • Generate performance metrics

Expected output:
  
  Training on STEAD...
  Epoch 1/50
    Train - Loss: 0.4523, Acc: 0.8234, F1: 0.8156
    Val   - Loss: 0.3845, Acc: 0.8567, F1: 0.8523
  
  ...
  
  Testing on DAS...
  ══════════════════════════════════════════════
  DAS Test Results:
  ══════════════════════════════════════════════
  Accuracy: 0.8750
  Precision: 0.8571
  Recall: 0.9000
  F1-Score: 0.8780

═══════════════════════════════════════════════════════════════════════

STEP 4: Compare with DAS-Only Training
───────────────────────────────────────

For comparison, train only on DAS:
  
  $ python main.py

This trains and tests using only your DAS dataset.

Compare results:
  • STEAD → DAS transfer learning
  • DAS-only training

Which performs better on your specific deployment?

═══════════════════════════════════════════════════════════════════════

Key Differences: STEAD vs DAS
──────────────────────────────

STEAD Dataset:
  • 1.2M waveforms
  • 3 channels (E, N, Z)
  • 6000 time samples @ 100Hz
  • Global earthquakes
  • Various instruments

DAS Dataset:
  • ~19 files (13 earthquakes, 6 quarry blasts)
  • 100-10,000 channels
  • 2000-6000 samples @ 200Hz
  • Local events (Bigorre, France)
  • Single fiber optic cable

═══════════════════════════════════════════════════════════════════════

How Channel Adaptation Works
─────────────────────────────

STEAD has 3 channels, DAS has many more (e.g., 2000).
The adapter creates synthetic spatial channels:

  1. Place 3 STEAD channels at even intervals
  2. Interpolate between them
  3. Add small noise for variation

This allows the model to learn from:
  • Temporal patterns (STEAD)
  • Spatial patterns (synthetic)

═══════════════════════════════════════════════════════════════════════

Configuration
─────────────

Edit config.yaml to customize:

  data:
    stead_window_size: 6000  # STEAD samples
    window_size: 2000        # DAS samples
    batch_size: 16
  
  training:
    epochs: 50
    learning_rate: 0.001
  
  model:
    type: "lstm"
    hidden_dim: 64

═══════════════════════════════════════════════════════════════════════

Monitoring Training
───────────────────

Use TensorBoard to visualize:
  
  $ tensorboard --logdir=logs
  $ open http://localhost:6006

You'll see:
  • Loss curves (training/validation)
  • Accuracy over time
  • F1-score trends

═══════════════════════════════════════════════════════════════════════

Understanding Results
─────────────────────

Good Transfer Learning (>80% accuracy):
  ✓ Model learned generalizable patterns
  ✓ STEAD training was beneficial
  ✓ Can work with limited DAS data

Poor Transfer Learning (<60% accuracy):
  ⚠ DAS data too different from STEAD
  ⚠ Need more DAS-specific training
  ⚠ Try fine-tuning approach

═══════════════════════════════════════════════════════════════════════

Fine-Tuning Approach
────────────────────

If transfer learning shows promise but needs improvement:

1. Train on STEAD (pre-training)
2. Load trained model
3. Continue training on DAS for a few epochs
4. This adapts general patterns to your specific deployment

(This requires modifying the training script)

═══════════════════════════════════════════════════════════════════════

Output Files
────────────

After training, you'll have:

  models/
    └── best_model_stead.pth        # Trained model
    └── confusion_matrix_das_test.png  # Visualization
    └── das_test_results.json       # Metrics
  
  logs/
    └── events.*                    # TensorBoard logs

═══════════════════════════════════════════════════════════════════════

Next Steps
──────────

1. Run the training pipeline
2. Analyze results
3. Compare with DAS-only training
4. Experiment with:
   • Different STEAD filters (magnitude, distance)
   • Model architectures (try CNN)
   • Augmentation parameters
   • Learning rates

═══════════════════════════════════════════════════════════════════════

For detailed information, see:
  • STEAD_TRAINING_GUIDE.md - Complete guide
  • README_STEAD.md - STEAD dataset info
  • ARCHITECTURE.md - Model details

Questions? Check the troubleshooting section in STEAD_TRAINING_GUIDE.md

═══════════════════════════════════════════════════════════════════════
    """)


if __name__ == '__main__':
    example_workflow()
    
    print("\nReady to start?")
    print("  1. Setup: python setup_stead.py")
    print("  2. Train:  python train_stead_test_das.py")
    print()
