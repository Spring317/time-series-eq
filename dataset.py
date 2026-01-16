"""
Memory-efficient data loader for DAS seismic data
Uses lazy loading and memory mapping to handle large datasets
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import os
from pathlib import Path


class DASDataset(Dataset):
    """
    Memory-efficient dataset that loads DAS data on-demand
    Uses h5py with memory mapping to avoid loading entire dataset into RAM
    """
    
    def __init__(
        self,
        file_paths: List[str],
        labels,  # Can be List[int] or Dict[str, int]
        window_size: int = 1024,
        stride: int = 512,
        num_channels: Optional[int] = None,
        transform=None,
        cache_file_info: bool = True
    ):
        """
        Args:
            file_paths: List of paths to HDF5 files
            labels: List of labels or dict mapping filenames to labels (0=earthquake, 1=quarry blast)
            window_size: Number of time samples per window
            stride: Stride for sliding window
            num_channels: Number of DAS channels (auto-detect if None)
            transform: Optional data augmentation function
            cache_file_info: Cache file metadata to speed up initialization
        """
        self.file_paths = file_paths
        
        # Convert labels dict to list if necessary
        if isinstance(labels, dict):
            self.labels = []
            for file_path in file_paths:
                filename = Path(file_path).name
                if filename in labels:
                    self.labels.append(labels[filename])
                else:
                    raise ValueError(f"Label not found for file: {filename}")
        else:
            self.labels = labels
            
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.num_channels = num_channels
        
        # Build index of all windows across all files
        self.windows_index = []
        self._build_index(cache_file_info)
        
    def _build_index(self, use_cache: bool):
        """
        Build an index of all windows without loading data
        Each entry: (file_idx, start_idx, num_samples, num_channels)
        """
        # Create a unique cache key based on file paths and parameters
        import hashlib
        files_hash = hashlib.md5(
            '|'.join(self.file_paths).encode() + 
            f'{self.window_size}_{self.stride}'.encode()
        ).hexdigest()[:12]
        cache_path = Path(f"cache/dataset_index_{files_hash}.npy")
        
        if use_cache and cache_path.exists():
            print("Loading cached dataset index...")
            cached_data = np.load(cache_path, allow_pickle=True).item()
            self.windows_index = cached_data['windows_index']
            self.num_channels = cached_data.get('num_channels', self.num_channels)
            return
        
        print("Building dataset index...")
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                with h5py.File(file_path, 'r') as f:
                    # DAS-BIGORRE data structure: python_processing/sr/data contains the strain rate data
                    data_key = None
                    shape = None
                    
                    # Helper function to check nested paths
                    def check_path(f, path):
                        parts = path.split('/')
                        obj = f
                        for part in parts:
                            if part in obj:
                                obj = obj[part]
                            else:
                                return None
                        return obj
                    
                    # Try common paths (processed files use 'data' directly)
                    possible_paths = [
                        'data',
                        'python_processing/sr/data',
                        'sr/data'
                    ]
                    
                    for path in possible_paths:
                        try:
                            dataset = check_path(f, path)
                            if dataset is not None and hasattr(dataset, 'shape'):
                                shape = dataset.shape
                                data_key = path
                                break
                        except Exception as e:
                            continue
                    
                    if data_key is None or shape is None:
                        print(f"Warning: Cannot find data in {file_path}")
                        print(f"  Available keys: {list(f.keys())}")
                        continue
                    
                    # Shape could be (time, channels) or (channels, time)
                    # Detect based on which dimension is larger
                    if len(shape) == 2:
                        if shape[0] > shape[1]:
                            num_samples, num_channels = shape
                        else:
                            num_channels, num_samples = shape
                    else:
                        print(f"Warning: Unexpected shape {shape} in {file_path}")
                        continue
                    
                    if self.num_channels is None:
                        self.num_channels = num_channels
                        print(f"Setting reference channel count: {num_channels}")
                    elif num_channels != self.num_channels:
                        print(f"Warning: Channel mismatch in {Path(file_path).name}:")
                        print(f"  Expected {self.num_channels}, got {num_channels}")
                        print(f"  This file will be padded/cropped to match.")
                    
                    # Calculate number of windows in this file
                    num_windows = max(1, (num_samples - self.window_size) // self.stride + 1)
                    
                    for win_idx in range(num_windows):
                        start_idx = win_idx * self.stride
                        self.windows_index.append({
                            'file_idx': file_idx,
                            'start_idx': start_idx,
                            'data_key': data_key,
                            'label': self.labels[file_idx],
                            'is_transposed': shape[0] <= shape[1],  # True if (channels, time)
                            'file_channels': num_channels  # Store actual channel count
                        })
                        
            except Exception as e:
                import traceback
                print(f"Error processing {file_path}:")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
        
        print(f"Total windows: {len(self.windows_index)}")
        
        # Cache the index
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'windows_index': self.windows_index,
                'num_channels': self.num_channels
            }
            np.save(cache_path, cache_data)
    
    def __len__(self):
        return len(self.windows_index)
    
    def __getitem__(self, idx):
        """
        Load a single window on-demand
        Returns: (data, label) where data shape is (channels, time)
        """
        window_info = self.windows_index[idx]
        file_idx = window_info['file_idx']
        start_idx = window_info['start_idx']
        data_key = window_info['data_key']
        label = window_info['label']
        is_transposed = window_info['is_transposed']
        
        # Helper function to navigate nested HDF5 groups
        def get_dataset(f, path):
            parts = path.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            return obj
        
        # Open file and read only the required window
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            dataset = get_dataset(f, data_key)
            if is_transposed:
                # Shape: (channels, time)
                data = dataset[:, start_idx:start_idx + self.window_size]
                # CRITICAL: Immediately copy to break memory mapping
                data = np.array(data, dtype=np.float32, copy=True)
                # Ensure we have the full window
                if data.shape[1] < self.window_size:
                    pad_width = ((0, 0), (0, self.window_size - data.shape[1]))
                    data = np.pad(data, pad_width, mode='constant')
            else:
                # Shape: (time, channels)
                data = dataset[start_idx:start_idx + self.window_size, :]
                # CRITICAL: Immediately copy to break memory mapping
                data = np.array(data, dtype=np.float32, copy=True)
                if data.shape[0] < self.window_size:
                    pad_width = ((0, self.window_size - data.shape[0]), (0, 0))
                    data = np.pad(data, pad_width, mode='constant')
                # Transpose to (channels, time)
                data = data.T
        
        # Handle channel count mismatches
        file_channels = window_info.get('file_channels', self.num_channels)
        if data.shape[0] != self.num_channels:
            if data.shape[0] < self.num_channels:
                # Pad with zeros
                pad_channels = self.num_channels - data.shape[0]
                padding = np.zeros((pad_channels, data.shape[1]), dtype=np.float32)
                data = np.vstack([data, padding])
            else:
                # Crop to match
                data = data[:self.num_channels, :]
        
        # Apply normalization per channel (ensure float32 throughout)
        mean = np.mean(data, axis=1, keepdims=True, dtype=np.float32)
        std = np.std(data, axis=1, keepdims=True, dtype=np.float32) + np.float32(1e-8)
        data = (data - mean) / std
        
        # Ensure contiguous array in memory
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        # Apply augmentation if specified
        if self.transform is not None:
            data = self.transform(data)
        
        # Convert to torch tensor - create independent copy
        # Using .copy() on numpy array before torch conversion ensures no shared memory
        data_copy = np.array(data, dtype=np.float32, copy=True, order='C')
        data = torch.from_numpy(data_copy)
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label


class DataAugmentation:
    """
    Data augmentation for seismic waveforms
    """
    
    def __init__(
        self,
        noise_level: float = 0.05,
        time_shift_range: int = 50,
        amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    ):
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
        self.amplitude_scale_range = amplitude_scale_range
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations
        Args:
            data: shape (channels, time)
        Returns:
            augmented data (contiguous float32 array)
        """
        # Ensure input is contiguous
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        # Add Gaussian noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(*data.shape).astype(np.float32) * self.noise_level
            data = data + noise
        
        # Random amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.float32(np.random.uniform(*self.amplitude_scale_range))
            data = data * scale
        
        # Time shift (circular shift)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-self.time_shift_range, self.time_shift_range)
            data = np.roll(data, shift, axis=1)
        
        # Ensure output is contiguous
        return np.ascontiguousarray(data, dtype=np.float32)


def create_dataloaders(
    file_paths: List[str],
    labels: List[int],
    config: dict,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create memory-efficient data loaders
    """
    
    # Create augmentation transform for training
    augment = None
    if config['augmentation']['use_augmentation']:
        augment = DataAugmentation(
            noise_level=config['augmentation']['noise_level'],
            time_shift_range=config['augmentation']['time_shift_range'],
            amplitude_scale_range=config['augmentation']['amplitude_scale_range']
        )
    
    # Create datasets
    train_files = [file_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_dataset = DASDataset(
        train_files,
        train_labels,
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        num_channels=config['data']['num_channels'],
        transform=augment
    )
    
    val_files = [file_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_dataset = DASDataset(
        val_files,
        val_labels,
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        num_channels=config['data']['num_channels'],
        transform=None
    )
    
    test_files = [file_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_dataset = DASDataset(
        test_files,
        test_labels,
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        num_channels=config['data']['num_channels'],
        transform=None
    )
    
    # Create data loaders with memory-efficient settings
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
        num_workers=0,  # Use single process for test to avoid HDF5 memory-map issues
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
