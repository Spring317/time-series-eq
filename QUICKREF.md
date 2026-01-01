# Quick Reference Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install h5py pyyaml tqdm scikit-learn matplotlib seaborn tensorboard
```

## Quick Start

```bash
# 1. Label your data
python create_labels.py

# 2. Train the model
python main.py

# 3. Monitor training
tensorboard --logdir logs

# 4. Run inference
python inference.py --checkpoint models/best_model.pth --input data/
```

## Common Commands

### Training
```bash
# Standard training
python main.py

# Resume from checkpoint (edit config to load checkpoint)
# Modify train.py to add checkpoint loading logic
```

### Inference
```bash
# Single file
python inference.py --checkpoint models/best_model.pth --input data/file.h5

# All files in directory
python inference.py --checkpoint models/best_model.pth --input data/

# CPU mode
python inference.py --checkpoint models/best_model.pth --input data/ --device cpu
```

### Visualization
```bash
# Visualize sample data
python visualize.py data data/DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5

# Plot training history (after training)
python visualize.py history

# Analyze predictions
python visualize.py predictions
```

## Configuration Presets

### High Memory (GPU with 16GB+ VRAM)
```yaml
data:
  batch_size: 32
  window_size: 2048
  num_workers: 6

model:
  type: "transformer"
  hidden_dim: 256
  num_layers: 6
```

### Medium Memory (GPU with 8GB VRAM)
```yaml
data:
  batch_size: 16
  window_size: 1024
  num_workers: 4

model:
  type: "temporal_cnn"
  hidden_dim: 128
  num_layers: 4
```

### Low Memory (GPU with 4GB VRAM or CPU)
```yaml
data:
  batch_size: 4
  window_size: 512
  num_workers: 2

model:
  type: "temporal_cnn"
  hidden_dim: 64
  num_layers: 3
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size
batch_size: 8  # or 4

# Reduce model size
hidden_dim: 64
num_layers: 3

# Reduce window size
window_size: 512
```

### Slow Data Loading
```yaml
# Increase workers
num_workers: 6

# Reduce window overlap
stride: 1024  # no overlap
```

### Poor Accuracy
1. Check labels are correct
2. Increase training time
   ```yaml
   epochs: 100
   ```
3. Try different model
   ```yaml
   type: "lstm"  # or "transformer"
   ```
4. Enable augmentation
   ```yaml
   use_augmentation: true
   ```
5. Adjust learning rate
   ```yaml
   learning_rate: 0.0001  # lower
   ```

### Training Takes Too Long
1. Use smaller model
   ```yaml
   hidden_dim: 64
   num_layers: 3
   ```
2. Reduce data size
   - Use fewer files
   - Increase stride
   ```yaml
   stride: 1024
   ```
3. Use GPU
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Model Selection Guide

| Model | Speed | Memory | Accuracy | Best For |
|-------|-------|--------|----------|----------|
| **Temporal CNN** | Fast | Low | Good | General purpose, fast training |
| **LSTM** | Medium | Medium | Good | Sequential patterns |
| **Transformer** | Slow | High | Best | Complex patterns, large dataset |

## Performance Benchmarks

### Expected Training Time (on RTX 3090)
- Temporal CNN: 2-3 hours
- LSTM: 3-4 hours
- Transformer: 5-6 hours

### Expected Accuracy
- With good labels: 85-95%
- With noisy labels: 70-85%
- Random labels: ~50% (chance)

## Directory Structure After Training

```
.
├── models/
│   ├── best_model.pth              # 50-200 MB
│   ├── checkpoint_epoch_5.pth      # 50-200 MB each
│   ├── confusion_matrix.png
│   └── test_results.json
├── logs/
│   └── events.out.tfevents.*       # 10-50 MB
├── cache/
│   └── dataset_index.npy           # 1-10 MB
└── predictions.json                # 10-100 KB
```

## Useful Tricks

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Disk I/O
```bash
iotop
```

### Check Data Statistics
```python
import h5py
import numpy as np

with h5py.File('data/file.h5', 'r') as f:
    data = f[list(f.keys())[0]][:]
    print(f"Shape: {data.shape}")
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Std: {np.std(data):.3f}")
    print(f"Min: {np.min(data):.3f}")
    print(f"Max: {np.max(data):.3f}")
```

### Export Model for Deployment
```python
import torch
from models import create_model

# Load checkpoint
checkpoint = torch.load('models/best_model.pth')

# Create model
config = checkpoint['config']
model = create_model(config, num_channels=256)
model.load_state_dict(checkpoint['model_state_dict'])

# Export to TorchScript
model.eval()
example_input = torch.randn(1, 256, 1024)
traced = torch.jit.trace(model, example_input)
traced.save('models/model_traced.pt')
```

## Best Practices

1. **Always label data carefully** - Model quality depends on label quality
2. **Use validation set** - Don't tune hyperparameters on test set
3. **Monitor training** - Use TensorBoard to catch problems early
4. **Save checkpoints** - Training can be interrupted
5. **Test on holdout data** - Ensure model generalizes
6. **Version your experiments** - Keep track of what works
7. **Start simple** - Use Temporal CNN first, then try others
8. **Check data first** - Visualize samples before training

## Common Errors

### ImportError: No module named 'h5py'
```bash
pip install h5py
```

### RuntimeError: CUDA out of memory
```yaml
# In config.yaml
batch_size: 4
```

### ValueError: No data in file
- Check HDF5 file is not corrupted
- Verify file has data: `h5ls data/file.h5`

### KeyError: 'model_state_dict'
- Checkpoint file might be corrupted
- Train again from scratch

## Contact & Support

For issues or questions:
1. Check this guide first
2. Review TRAINING_README.md
3. Check ARCHITECTURE.md for technical details
4. Open an issue on GitHub
