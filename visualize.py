"""
Utility script to visualize data and training progress
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def visualize_sample_data(file_path, max_channels=50, max_time=1000):
    """
    Visualize a sample of DAS data
    """
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if len(keys) == 0:
            print("No data in file")
            return
        
        data_key = keys[0]
        data = f[data_key][:]
        
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Data range: [{np.min(data)}, {np.max(data)}]")
        
        # Ensure correct orientation (channels, time)
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        # Subsample for visualization
        data = data[:max_channels, :max_time]
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Waterfall plot
        axes[0].imshow(data, aspect='auto', cmap='seismic', 
                      interpolation='nearest', vmin=-np.percentile(np.abs(data), 95),
                      vmax=np.percentile(np.abs(data), 95))
        axes[0].set_title('DAS Data (Waterfall Plot)')
        axes[0].set_xlabel('Time Samples')
        axes[0].set_ylabel('Channel')
        axes[0].set_colorbar()
        
        # Single channel trace
        channel_idx = data.shape[0] // 2
        axes[1].plot(data[channel_idx, :], linewidth=0.5)
        axes[1].set_title(f'Single Channel Trace (Channel {channel_idx})')
        axes[1].set_xlabel('Time Samples')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_data_visualization.png', dpi=150)
        print("\n✓ Saved visualization to sample_data_visualization.png")
        plt.close()


def plot_training_history(log_file='logs/training_history.json'):
    """
    Plot training history from saved logs
    """
    if not Path(log_file).exists():
        print(f"Log file {log_file} not found")
        return
    
    with open(log_file, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], label='Validation', linewidth=2)
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], linewidth=2)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✓ Saved training history to training_history.png")
    plt.close()


def analyze_predictions(predictions_file='predictions.json'):
    """
    Analyze prediction results
    """
    if not Path(predictions_file).exists():
        print(f"Predictions file {predictions_file} not found")
        return
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Count predictions
    class_counts = {}
    confidences = []
    
    for pred in predictions:
        class_name = pred['class_name']
        confidence = pred['confidence']
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(confidence)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Class distribution
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    axes[0].bar(classes, counts, color=['#3498db', '#e74c3c'])
    axes[0].set_title('Predicted Class Distribution')
    axes[0].set_ylabel('Count')
    
    # Confidence distribution
    axes[1].hist(confidences, bins=20, color='#2ecc71', edgecolor='black')
    axes[1].set_title('Prediction Confidence Distribution')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(confidences):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=150)
    print("✓ Saved prediction analysis to prediction_analysis.png")
    plt.close()
    
    print(f"\nPrediction Summary:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py data <file.h5>      - Visualize DAS data")
        print("  python visualize.py history             - Plot training history")
        print("  python visualize.py predictions         - Analyze predictions")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'data':
        if len(sys.argv) < 3:
            print("Please provide HDF5 file path")
            sys.exit(1)
        visualize_sample_data(sys.argv[2])
    
    elif command == 'history':
        plot_training_history()
    
    elif command == 'predictions':
        analyze_predictions()
    
    else:
        print(f"Unknown command: {command}")
