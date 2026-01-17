"""
Load pre-processed STEAD data from numpy files

These files were processed with LightEQ preprocessing (STFT spectrograms)
Expected shape: X = (N, 151, 41, 3), y = (N, 76, 1) or (N,)

Memory-efficient loading using memory-mapped files
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional


class MemoryEfficientSTEADDataset(Dataset):
    """
    Memory-efficient STEAD dataset using numpy memory-mapping
    
    Loads data directly from disk without loading entire array into RAM
    """
    
    def __init__(
        self,
        X_path: str,
        y_path: str,
        indices: np.ndarray = None,
        transform_labels: bool = True
    ):
        """
        Args:
            X_path: Path to X numpy file
            y_path: Path to y numpy file
            indices: Optional subset of indices to use
            transform_labels: If True, convert detection labels to classification
        """
        # Memory-map the arrays (read-only)
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        
        self.indices = indices if indices is not None else np.arange(len(self.X))
        self.transform_labels = transform_labels
        
        # Check shapes
        print(f"  X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"  Using {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Load single sample (copies from mmap to memory)
        x = np.array(self.X[real_idx], dtype=np.float32)
        y = np.array(self.y[real_idx])
        
        # Convert from (151, 41, 3) to (3, 151, 41) for PyTorch
        if x.ndim == 3 and x.shape[-1] == 3:
            x = np.transpose(x, (2, 0, 1))
        
        # Handle detection labels (N, 76, 1) or (N, 76) -> classification label
        if self.transform_labels and y.ndim >= 1 and y.shape[0] > 1:
            # If any time step has earthquake (1), classify as earthquake (0)
            # Otherwise classify as noise (1)
            y_flat = y.flatten()
            label = 0 if np.any(y_flat > 0.5) else 1
        else:
            label = int(y.flatten()[0]) if y.size > 0 else 0
        
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def create_dataloaders_from_preprocessed(
    data_dir: str,
    prefix: str = "76",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    max_samples: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from pre-processed numpy files
    
    Memory-efficient: uses memory-mapped files
    
    Args:
        data_dir: Directory containing numpy files
        prefix: File prefix (e.g., "76")
        batch_size: Batch size
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers
        seed: Random seed for validation split
        max_samples: Maximum samples to use (for memory/speed)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    
    X_train_path = data_dir / f"X_train{prefix}.npy"
    y_train_path = data_dir / f"y_train{prefix}.npy"
    X_test_path = data_dir / f"X_test{prefix}.npy"
    y_test_path = data_dir / f"y_test{prefix}.npy"
    
    print(f"Loading pre-processed STEAD data from {data_dir}...")
    
    # Get total samples without loading full array
    X_train_shape = np.load(X_train_path, mmap_mode='r').shape
    X_test_shape = np.load(X_test_path, mmap_mode='r').shape
    
    print(f"  X_train shape: {X_train_shape}")
    print(f"  X_test shape: {X_test_shape}")
    
    n_train_total = X_train_shape[0]
    n_test = X_test_shape[0]
    
    # Create indices
    np.random.seed(seed)
    all_train_indices = np.random.permutation(n_train_total)
    
    # Limit samples if requested
    if max_samples and max_samples < n_train_total:
        all_train_indices = all_train_indices[:max_samples]
        print(f"  Limited to {max_samples} training samples")
    
    # Split into train/val
    n_total = len(all_train_indices)
    n_val = int(n_total * val_split)
    
    val_indices = all_train_indices[:n_val]
    train_indices = all_train_indices[n_val:]
    
    # Test indices (optionally limit)
    test_indices = np.arange(n_test)
    if max_samples and max_samples < n_test:
        test_indices = test_indices[:max_samples // 10]  # Use 10% of max for test
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")
    
    # Create datasets (memory-efficient)
    print("\nCreating memory-mapped datasets...")
    train_dataset = MemoryEfficientSTEADDataset(
        str(X_train_path), str(y_train_path), train_indices
    )
    val_dataset = MemoryEfficientSTEADDataset(
        str(X_train_path), str(y_train_path), val_indices
    )
    test_dataset = MemoryEfficientSTEADDataset(
        str(X_test_path), str(y_test_path), test_indices
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader


# Keep old functions for backward compatibility
def load_preprocessed_stead(
    data_dir: str,
    prefix: str = "76"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-processed STEAD numpy files (loads all into memory - use with caution!)
    
    For large datasets, use create_dataloaders_from_preprocessed() instead.
    """
    data_dir = Path(data_dir)
    
    print(f"WARNING: Loading entire dataset into memory...")
    print(f"For large datasets, use create_dataloaders_from_preprocessed() instead")
    
    X_train = np.load(data_dir / f"X_train{prefix}.npy")
    y_train = np.load(data_dir / f"y_train{prefix}.npy")
    X_test = np.load(data_dir / f"X_test{prefix}.npy")
    y_test = np.load(data_dir / f"y_test{prefix}.npy")
    
    return X_train, y_train, X_test, y_test


class PreprocessedSTEADDataset(Dataset):
    """Legacy dataset - loads from pre-loaded numpy arrays"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 4 and X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))
        
        self.X = torch.from_numpy(X.astype(np.float32))
        
        # Handle multi-dimensional labels
        if y.ndim > 1:
            # Convert detection labels to classification
            y_class = np.array([0 if np.any(row > 0.5) else 1 for row in y.reshape(len(y), -1)])
            self.y = torch.from_numpy(y_class.astype(np.int64))
        else:
            self.y = torch.from_numpy(y.astype(np.int64))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    import sys
    
    stead_dir = "/storage/student8/STEAD"
    if len(sys.argv) > 1:
        stead_dir = sys.argv[1]
    
    if not Path(stead_dir).exists():
        print(f"Directory not found: {stead_dir}")
        sys.exit(1)
    
    # Test memory-efficient loading
    train_loader, val_loader, test_loader = create_dataloaders_from_preprocessed(
        stead_dir,
        prefix="76",
        batch_size=64,
        max_samples=100000  # Limit for testing
    )
    
    # Test a batch
    for X, y in train_loader:
        print(f"\nBatch shape: {X.shape}")
        print(f"Labels: {y[:10]}")
        print(f"Label distribution: {torch.bincount(y)}")
        break
