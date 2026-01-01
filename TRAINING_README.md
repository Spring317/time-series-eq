# DAS Seismic Event Classification - Deep Learning Pipeline

Memory-efficient training pipeline for classifying seismic events (Earthquakes vs Quarry Blasts) using DAS (Distributed Acoustic Sensing) data with deep learning models.

## 🚀 Features

- **Memory-efficient data loading**: Handles 300GB+ datasets with only 32GB RAM using lazy loading and HDF5 memory mapping
- **Multiple model architectures**: Choose from Temporal CNN, LSTM, or Transformer models
- **Data augmentation**: Gaussian noise, amplitude scaling, and time shifting
- **Comprehensive metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **TensorBoard integration**: Real-time training visualization
- **Early stopping**: Automatic training termination to prevent overfitting
- **Checkpointing**: Save and resume training

## 📋 Requirements

- Python 3.8+
- 32GB RAM (minimum)
- GPU with CUDA support (recommended)
- ~1GB disk space for models and logs

## 🔧 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Structure

Your data should be organized as follows:
```
data/
├── DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5
├── DAS-BIGORRE_2022-09-01_09-19-21_UTC-016.h5
└── ...
```

HDF5 files should contain seismic waveform data with shape:
- `(time_samples, channels)` or `(channels, time_samples)`
- The code automatically detects the correct orientation

## 🏷️ Data Labeling

Before training, you need to label your data files. Use the interactive labeling tool:

```bash
python create_labels.py
```

This will create a `labels.json` file:
```json
{
  "DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5": 0,
  "DAS-BIGORRE_2022-09-01_09-19-21_UTC-016.h5": 1,
  ...
}
```

Labels:
- `0` = Earthquake
- `1` = Quarry Blast

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters:

```yaml
data:
  window_size: 1024        # Time samples per window
  stride: 512              # Overlap between windows
  batch_size: 16           # Adjust based on GPU memory
  num_channels: 256        # Number of DAS channels

training:
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10

model:
  type: "temporal_cnn"     # Options: temporal_cnn, lstm, transformer
  hidden_dim: 128
  num_layers: 4
  dropout: 0.3
```

### Memory Management Tips

If you run out of memory:
1. Reduce `batch_size` (e.g., 8 or 4)
2. Reduce `num_workers` (e.g., 2)
3. Reduce `window_size` (e.g., 512)
4. Use smaller model (reduce `hidden_dim` or `num_layers`)

## 🎯 Training

Run the training pipeline:

```bash
python main.py
```

The training process will:
1. Load and validate your labeled data
2. Split into train/validation/test sets (70/20/10)
3. Train the model with automatic checkpointing
4. Evaluate on the test set
5. Generate confusion matrix and metrics

### Monitor Training

View training progress in real-time:

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## 📈 Model Architectures

### 1. Temporal CNN (Default)
- Best for: Fast training, good performance
- Memory: Low
- Speed: Fast

### 2. LSTM
- Best for: Sequential patterns, temporal dependencies
- Memory: Medium
- Speed: Medium

### 3. Transformer
- Best for: Long-range dependencies, complex patterns
- Memory: High
- Speed: Slow

## 🔮 Inference

Classify new seismic data using a trained model:

```bash
# Single file
python inference.py \
  --checkpoint models/best_model.pth \
  --input data/new_event.h5 \
  --output predictions.json

# Directory of files
python inference.py \
  --checkpoint models/best_model.pth \
  --input data/ \
  --output predictions.json
```

## 📁 Output Structure

After training, you'll have:

```
models/
├── best_model.pth              # Best model checkpoint
├── checkpoint_epoch_5.pth      # Periodic checkpoints
├── confusion_matrix.png        # Visualization
└── test_results.json           # Final metrics

logs/
└── events.out.tfevents.*       # TensorBoard logs

cache/
└── dataset_index.npy           # Cached dataset index
```

## 🎓 Example Workflow

1. **Prepare data**:
```bash
python create_labels.py
```

2. **Train model**:
```bash
python main.py
```

3. **View training**:
```bash
tensorboard --logdir logs
```

4. **Classify new data**:
```bash
python inference.py --checkpoint models/best_model.pth --input data/
```

## 🛠️ Memory Optimization Techniques

This pipeline uses several techniques to handle large datasets efficiently:

1. **Lazy Loading**: Data is loaded on-demand, not preloaded into RAM
2. **HDF5 Memory Mapping**: Direct file access without full loading
3. **Sliding Window**: Processes data in small chunks
4. **Batch Processing**: Efficient GPU utilization
5. **Index Caching**: Fast dataset initialization
6. **Persistent Workers**: Reduces data loading overhead

## 📊 Expected Performance

With proper labeling, you should achieve:
- **Accuracy**: 85-95%
- **F1-Score**: 85-95%
- **Training time**: 2-6 hours (depends on GPU and dataset size)

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.yaml
batch_size: 8  # or even 4
```

### Slow Training
```bash
# Increase num_workers in config.yaml
num_workers: 4  # adjust based on CPU cores
```

### Poor Performance
- Check data labels are correct
- Try different model architecture
- Increase training epochs
- Adjust learning rate
- Use data augmentation

## 📝 File Structure

```
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── create_labels.py         # Interactive labeling tool
├── dataset.py              # Memory-efficient data loader
├── models.py               # Model architectures (CNN, LSTM, Transformer)
├── train.py                # Training logic
├── main.py                 # Main training script
├── inference.py            # Inference script
├── data/                   # DAS HDF5 files
├── models/                 # Saved models
├── logs/                   # TensorBoard logs
└── cache/                  # Dataset index cache
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
