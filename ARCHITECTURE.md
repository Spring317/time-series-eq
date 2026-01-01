# Training Pipeline Architecture

## Overall Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DAS Data (300GB)                             │
│                    19 HDF5 files in data/                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Labeling (create_labels.py)                │
│   Interactively label files: 0=Earthquake, 1=Quarry Blast           │
│                    Output: labels.json                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Memory-Efficient Data Loader                      │
│                        (dataset.py)                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Lazy Loading: HDF5 files opened on-demand                 │  │
│  │  2. Sliding Window: Break into 1024-sample windows            │  │
│  │  3. Index Caching: Fast dataset initialization                │  │
│  │  4. Per-channel Normalization                                 │  │
│  │  5. Data Augmentation (training only)                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │  Train  │   │   Val   │   │  Test   │
              │  (70%)  │   │  (20%)  │   │  (10%)  │
              └─────────┘   └─────────┘   └─────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Model Architecture                            │
│                          (models.py)                                 │
│                                                                       │
│  Option 1: Temporal CNN                                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Input (B, C, T) → Channel Projection → Conv Blocks →         │  │
│  │  Global Pool → Classifier → Output (B, 2)                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Option 2: LSTM                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Input (B, C, T) → Input Projection → BiLSTM →                │  │
│  │  Hidden State → Classifier → Output (B, 2)                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Option 3: Transformer                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Input (B, C, T) → Input Projection + Pos Encoding →          │  │
│  │  Transformer → CLS Token → Classifier → Output (B, 2)         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Loop (train.py)                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  For each epoch:                                               │  │
│  │    1. Forward pass (batch by batch)                           │  │
│  │    2. Calculate loss (CrossEntropy with class weights)        │  │
│  │    3. Backward pass + gradient clipping                       │  │
│  │    4. Optimizer step (AdamW)                                  │  │
│  │    5. Validation                                              │  │
│  │    6. Save best model                                         │  │
│  │    7. Early stopping check                                    │  │
│  │    8. Clear GPU cache                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Monitoring & Logging                            │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │   TensorBoard       │  │   Model Checkpoints │                   │
│  │   - Loss curves     │  │   - best_model.pth  │                   │
│  │   - Accuracy        │  │   - epoch_*.pth     │                   │
│  │   - F1 scores       │  │                     │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Evaluation & Results                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  - Confusion Matrix                                            │  │
│  │  - Classification Report                                       │  │
│  │  - ROC-AUC Score                                              │  │
│  │  - Precision, Recall, F1                                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Inference (inference.py)                        │
│  Load trained model → Process new data → Generate predictions        │
└─────────────────────────────────────────────────────────────────────┘
```

## Memory Optimization Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                     300GB Dataset → 32GB RAM                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │                           │
                    ▼                           ▼
        ┌────────────────────┐      ┌────────────────────┐
        │  Lazy Loading      │      │  Sliding Window    │
        │  - HDF5 mmap       │      │  - 1024 samples    │
        │  - On-demand read  │      │  - 512 stride      │
        └────────────────────┘      └────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │  Batch Processing          │
                    │  - Small batches (16)      │
                    │  - 4 workers               │
                    │  - Pin memory              │
                    └────────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Memory Management         │
                    │  - Gradient accumulation   │
                    │  - Cache clearing          │
                    │  - Efficient data types    │
                    └────────────────────────────┘
```

## Data Flow Per Window

```
Raw HDF5 Data
     │
     ▼
Extract Window (1024 samples × N channels)
     │
     ▼
Normalize (per-channel: mean=0, std=1)
     │
     ▼
Augmentation (if training)
  ├─ Add Gaussian noise (5%)
  ├─ Scale amplitude (0.9-1.1x)
  └─ Time shift (±50 samples)
     │
     ▼
Convert to Tensor (float32)
     │
     ▼
Model Input: (Batch, Channels, Time)
     │
     ▼
Model Output: (Batch, 2) logits
     │
     ▼
Softmax → Class Probabilities
     │
     ▼
Argmax → Predicted Class (0 or 1)
```

## File Structure

```
seis-learning-spatial/
│
├── config.yaml              ← Configuration
├── labels.json              ← File labels (created by user)
│
├── create_labels.py         ← Labeling tool
├── dataset.py              ← Data loader (memory-efficient)
├── models.py               ← Model architectures
├── train.py                ← Training logic
├── main.py                 ← Main entry point
├── inference.py            ← Inference script
├── visualize.py            ← Visualization utilities
├── setup.sh                ← Setup script
│
├── data/                   ← HDF5 files (300GB)
│   ├── *.h5
│   └── ...
│
├── cache/                  ← Dataset index cache
│   └── dataset_index.npy
│
├── models/                 ← Saved models
│   ├── best_model.pth
│   ├── checkpoint_*.pth
│   ├── confusion_matrix.png
│   └── test_results.json
│
└── logs/                   ← TensorBoard logs
    └── events.out.tfevents.*
```

## Key Features for Memory Efficiency

1. **No Full Dataset Loading**: Never load entire dataset into RAM
2. **HDF5 Memory Mapping**: Direct file access without copying
3. **Sliding Window**: Process small chunks (1024 samples)
4. **Batch Size Control**: Small batches (16) fit in GPU memory
5. **Data Workers**: Parallel data loading (4 workers)
6. **Index Caching**: Pre-compute window locations
7. **On-Demand Reading**: Load only what's needed for current batch
8. **Persistent Workers**: Reduce worker initialization overhead
9. **Pin Memory**: Faster CPU→GPU transfer
10. **Cache Clearing**: Periodic GPU memory cleanup
