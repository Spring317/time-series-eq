"""
STEAD (STanford EArthquake Dataset) loader for seismic classification
Adapted to work with the DAS classification model
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path


class STEADDataset(Dataset):
    """
    STEAD dataset loader for earthquake waveforms
    Each STEAD sample has 3 channels (E, N, Z) and 6000 time samples
    We adapt this to work with the same model architecture as DAS
    """
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        window_size: int = 6000,
        target_channels: Optional[int] = None,
        transform=None,
        filter_params: dict = None
    ):
        """
        Args:
            hdf5_path: Path to STEAD HDF5 file (e.g., 'merge.hdf5')
            csv_path: Path to STEAD CSV metadata file
            window_size: Number of time samples (STEAD has 6000)
            target_channels: Target number of channels (for compatibility with DAS model)
                           If None, uses 3 channels. If > 3, channels are replicated/padded
            transform: Optional data augmentation function
            filter_params: Dict with filtering criteria, e.g.:
                          {'trace_category': 'earthquake_local', 
                           'source_distance_km': ('<=', 200),
                           'source_magnitude': ('>', 2.5)}
        """
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.target_channels = target_channels
        self.transform = transform
        
        # Load and filter metadata
        print(f"Loading STEAD metadata from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Total traces in CSV: {len(df)}")
        
        # Apply filters
        if filter_params:
            for key, value in filter_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    op, threshold = value
                    if op == '<=':
                        df = df[df[key] <= threshold]
                    elif op == '>=':
                        df = df[df[key] >= threshold]
                    elif op == '<':
                        df = df[df[key] < threshold]
                    elif op == '>':
                        df = df[df[key] > threshold]
                    elif op == '==':
                        df = df[df[key] == threshold]
                else:
                    df = df[df[key] == value]
        
        self.trace_names = df['trace_name'].tolist()
        print(f"Traces after filtering: {len(self.trace_names)}")
        
        # For binary classification, we'll use:
        # 0 = earthquake, 1 = noise
        # (matching DAS labels: 0=earthquake, 1=quarry blast)
        self.labels = []
        for _, row in df.iterrows():
            if row['trace_category'] == 'noise':
                self.labels.append(1)
            else:  # earthquake_local, earthquake_regional
                self.labels.append(0)
        
        print(f"Label distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = 'Earthquake' if label == 0 else 'Noise'
            print(f"  {label_name} ({label}): {count}")
        
    def __len__(self):
        return len(self.trace_names)
    
    def __getitem__(self, idx):
        """
        Load a single STEAD waveform
        Returns: (data, label) where data shape is (channels, time)
        """
        trace_name = self.trace_names[idx]
        label = self.labels[idx]
        
        # Load waveform from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f.get(f'data/{trace_name}')
            if dataset is None:
                raise ValueError(f"Trace {trace_name} not found in HDF5 file")
            
            # STEAD format: (6000, 3) - (time, channels)
            # We need: (channels, time)
            data = np.array(dataset, dtype=np.float32)
            
            # Handle different possible shapes
            if len(data.shape) == 2:
                if data.shape[0] > data.shape[1]:
                    # (time, channels) -> transpose to (channels, time)
                    data = data.T
                # else already (channels, time)
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Ensure correct window size
        if data.shape[1] < self.window_size:
            # Pad if too short
            pad_width = ((0, 0), (0, self.window_size - data.shape[1]))
            data = np.pad(data, pad_width, mode='constant')
        elif data.shape[1] > self.window_size:
            # Crop if too long
            data = data[:, :self.window_size]
        
        # Handle channel adaptation for DAS compatibility
        # STEAD has 3 channels (E, N, Z), DAS might have 100s or 1000s
        if self.target_channels is not None and self.target_channels != data.shape[0]:
            data = self._adapt_channels(data, self.target_channels)
        
        # Normalize per channel
        mean = np.mean(data, axis=1, keepdims=True, dtype=np.float32)
        std = np.std(data, axis=1, keepdims=True, dtype=np.float32) + np.float32(1e-8)
        data = (data - mean) / std
        
        # Ensure contiguous array
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        # Apply augmentation if specified
        if self.transform is not None:
            data = self.transform(data)
        
        # Convert to torch tensor
        data_copy = np.array(data, dtype=np.float32, copy=True, order='C')
        data = torch.from_numpy(data_copy)
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label
    
    def _adapt_channels(self, data: np.ndarray, target_channels: int) -> np.ndarray:
        """
        Adapt 3-channel STEAD data to match target channel count
        
        Strategy:
        - If target < 3: Use only first N channels
        - If target == 3: Use as-is
        - If target > 3: Replicate channels with noise to simulate spatial sensors
        """
        current_channels = data.shape[0]
        
        if target_channels < current_channels:
            # Use subset of channels
            return data[:target_channels, :]
        
        elif target_channels == current_channels:
            return data
        
        else:
            # Replicate and add noise to simulate spatial distribution
            # This creates synthetic "spatial" channels from the 3 seismic channels
            time_samples = data.shape[1]
            adapted = np.zeros((target_channels, time_samples), dtype=np.float32)
            
            # Distribute the 3 real channels evenly
            spacing = target_channels // 3
            positions = [0, spacing, 2 * spacing]
            
            # Place original channels
            for i, pos in enumerate(positions[:current_channels]):
                adapted[pos] = data[i]
            
            # Fill gaps by interpolating
            for i in range(target_channels):
                if i not in positions:
                    # Find nearest placed channels
                    left_idx = max([p for p in positions if p < i] + [-1])
                    right_idx = min([p for p in positions if p > i] + [target_channels])
                    
                    if left_idx >= 0 and right_idx < target_channels:
                        # Interpolate between left and right
                        weight = (i - left_idx) / (right_idx - left_idx)
                        adapted[i] = (1 - weight) * adapted[left_idx] + weight * adapted[right_idx]
                        # Add small noise for variation
                        noise_level = 0.05
                        adapted[i] += np.random.randn(time_samples).astype(np.float32) * noise_level
                    elif left_idx >= 0:
                        # Extend from left
                        adapted[i] = adapted[left_idx] + np.random.randn(time_samples).astype(np.float32) * 0.05
                    elif right_idx < target_channels:
                        # Extend from right
                        adapted[i] = adapted[right_idx] + np.random.randn(time_samples).astype(np.float32) * 0.05
            
            return adapted


def create_stead_dataloaders(
    hdf5_path: str,
    csv_path: str,
    config: dict,
    target_channels: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    filter_params: dict = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from STEAD dataset
    
    Args:
        hdf5_path: Path to STEAD HDF5 file
        csv_path: Path to STEAD CSV file
        config: Configuration dict
        target_channels: Target number of channels (for DAS compatibility)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        filter_params: Filtering criteria for STEAD data
    """
    from dataset import DataAugmentation
    
    # Create augmentation for training
    augment = None
    if config['augmentation']['use_augmentation']:
        augment = DataAugmentation(
            noise_level=config['augmentation']['noise_level'],
            time_shift_range=config['augmentation']['time_shift_range'],
            amplitude_scale_range=config['augmentation']['amplitude_scale_range']
        )
    
    # Create full dataset
    full_dataset = STEADDataset(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        window_size=config['data'].get('stead_window_size', 6000),
        target_channels=target_channels,
        transform=None,  # We'll apply augmentation only to training split
        filter_params=filter_params
    )
    
    # Split into train/val/test
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Use random split with fixed seed for reproducibility
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Wrap training dataset with augmentation
    if augment is not None:
        class AugmentedDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                data, label = self.base_dataset[idx]
                if self.transform is not None:
                    # Convert to numpy, augment, convert back
                    data_np = data.numpy()
                    data_np = self.transform(data_np)
                    data = torch.from_numpy(data_np)
                return data, label
        
        train_dataset = AugmentedDataset(train_dataset, augment)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    return train_loader, val_loader, test_loader
