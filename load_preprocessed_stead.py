"""
Load pre-processed STEAD data from numpy files

These files were processed with LightEQ preprocessing (STFT spectrograms)
Expected shape: X = (N, 151, 41, 3), y = (N,) or (N, 1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional


def load_preprocessed_stead(
    data_dir: str,
    prefix: str = "76"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-processed STEAD numpy files
    
    Args:
        data_dir: Directory containing the numpy files
        prefix: File prefix (e.g., "76" for X_train76.npy)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    data_dir = Path(data_dir)
    
    # Load files
    print(f"Loading pre-processed STEAD data from {data_dir}...")
    
    X_train = np.load(data_dir / f"X_train{prefix}.npy")
    y_train = np.load(data_dir / f"y_train{prefix}.npy")
    X_test = np.load(data_dir / f"X_test{prefix}.npy")
    y_test = np.load(data_dir / f"y_test{prefix}.npy")
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Ensure labels are 1D
    if y_train.ndim > 1:
        y_train = y_train.squeeze()
    if y_test.ndim > 1:
        y_test = y_test.squeeze()
    
    # Check data format - LightEQ uses (N, time, freq, channels)
    # PyTorch needs (N, channels, time, freq)
    if X_train.ndim == 4:
        if X_train.shape[-1] == 3:  # (N, 151, 41, 3) -> need transpose
            print("Converting from (N, T, F, C) to (N, C, T, F) for PyTorch...")
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_test = np.transpose(X_test, (0, 3, 1, 2))
        elif X_train.shape[1] == 3:  # Already (N, 3, T, F)
            print("Data already in PyTorch format (N, C, T, F)")
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}, unique: {np.unique(y_train)}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}, unique: {np.unique(y_test)}")
    
    return X_train, y_train, X_test, y_test


class PreprocessedSTEADDataset(Dataset):
    """
    PyTorch Dataset for pre-processed STEAD data
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Spectrogram data (N, C, T, F) or (N, T, F, C)
            y: Labels (N,)
        """
        # Convert to PyTorch format if needed
        if X.ndim == 4 and X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))
        
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        
        if self.y.ndim > 1:
            self.y = self.y.squeeze()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders_from_preprocessed(
    data_dir: str,
    prefix: str = "76",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from pre-processed numpy files
    
    Args:
        data_dir: Directory containing numpy files
        prefix: File prefix (e.g., "76")
        batch_size: Batch size
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers
        seed: Random seed for validation split
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data
    X_train, y_train, X_test, y_test = load_preprocessed_stead(data_dir, prefix)
    
    # Split training into train/val
    np.random.seed(seed)
    n_train = len(X_train)
    n_val = int(n_train * val_split)
    
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = PreprocessedSTEADDataset(X_train, y_train)
    val_dataset = PreprocessedSTEADDataset(X_val, y_val)
    test_dataset = PreprocessedSTEADDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test loading
    import sys
    
    # Default path - update this to your STEAD location
    stead_dir = "/storage/student8/STEAD"
    
    if len(sys.argv) > 1:
        stead_dir = sys.argv[1]
    
    if not Path(stead_dir).exists():
        print(f"Directory not found: {stead_dir}")
        print("Usage: python load_preprocessed_stead.py /path/to/stead")
        sys.exit(1)
    
    train_loader, val_loader, test_loader = create_dataloaders_from_preprocessed(
        stead_dir,
        prefix="76",
        batch_size=64
    )
    
    # Test a batch
    for X, y in train_loader:
        print(f"\nBatch shape: {X.shape}")
        print(f"Labels: {y[:10]}")
        break
