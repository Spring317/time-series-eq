"""
STEAD Dataset with LightEQ State-of-the-Art Preprocessing
Implements STFT-based spectrogram conversion and augmentation from LightEQ paper

Reference:
LightEQ: On-Device Earthquake Detection with Embedded Machine Learning
Based on code by mostafamousavi, modified for LightEQ by TayyabaZainab0807
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path
from scipy import signal
from tqdm import tqdm


def normalize(data: np.ndarray, mode: str = 'max') -> np.ndarray:
    """
    Normalize waveform data (LightEQ style)
    
    Args:
        data: shape (time, channels)
        mode: 'max' or 'std' normalization
    """
    data = data.astype(np.float32)
    data -= np.mean(data, axis=0, keepdims=True)
    
    if mode == 'max':
        max_data = np.max(np.abs(data), axis=0, keepdims=True)
        max_data[max_data == 0] = 1  # Avoid division by zero
        data /= max_data
    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    
    return data


def shift_event(data: np.ndarray, addp: int, adds: int, coda_end: int, 
                snr: np.ndarray, rate: float) -> Tuple:
    """
    Shift the seismic event within the window (LightEQ augmentation)
    """
    org_len = len(data)
    
    if np.random.uniform(0, 1) < rate and snr is not None and all(snr >= 5.0):
        space = int(org_len - coda_end)
        preNoise = int(addp) - 100
        
        if preNoise > 0:
            noise0 = data[:preNoise, :]
            noise1 = noise0.copy()
            
            repN = int(np.floor(space / preNoise)) - 1
            if repN >= 1:
                for _ in range(min(repN, 5)):
                    noise1 = np.concatenate([noise1, noise0], axis=0)
                
                data2 = np.concatenate([noise1, data], axis=0)
                data2 = data2[:org_len, :]
                
                addp2 = addp + len(noise1)
                adds2 = adds + len(noise1)
                coda_end2 = min(coda_end + len(noise1), org_len)
                
                if 0 <= addp2 < org_len and 0 <= adds2 < org_len:
                    return data2, addp2, adds2, coda_end2
    
    return data, addp, adds, coda_end


def add_noise(data: np.ndarray, snr: np.ndarray, rate: float) -> np.ndarray:
    """
    Add Gaussian noise to waveform (LightEQ augmentation)
    """
    if np.random.uniform(0, 1) < rate and snr is not None and all(snr >= 5.0):
        data_noisy = np.empty_like(data)
        noise = np.random.normal(0, 1, data.shape[0])
        
        for ch in range(data.shape[1]):
            snr_factor = 10 ** (snr[ch] / 10) if ch < len(snr) else 1.0
            data_noisy[:, ch] = data[:, ch] + 0.5 * (noise * snr_factor) * np.random.random()
        
        return data_noisy.astype(np.float32)
    
    return data


def add_event(data: np.ndarray, addp: int, adds: int, coda_end: int, 
              snr: np.ndarray, rate: float) -> Tuple:
    """
    Add a second event to the waveform (LightEQ augmentation)
    """
    if addp is None or adds is None:
        return data, None
    
    s_p = adds - addp
    
    if (np.random.uniform(0, 1) < rate and 
        snr is not None and all(snr >= 5.0) and 
        (data.shape[0] - s_p - 21 - coda_end) > 20):
        
        added = np.copy(data)
        secondEV_strt = np.random.randint(coda_end, data.shape[0] - s_p - 21)
        space = data.shape[0] - secondEV_strt
        
        for ch in range(data.shape[1]):
            added[secondEV_strt:secondEV_strt+space, ch] += \
                data[addp:addp+space, ch] * np.random.uniform(0, 1)
        
        spt_secondEV = secondEV_strt
        sst_secondEV = spt_secondEV + s_p if spt_secondEV + s_p + 21 <= data.shape[0] else None
        
        if spt_secondEV and sst_secondEV:
            return added, [spt_secondEV, sst_secondEV]
    
    return data, None


def scale_amplitude(data: np.ndarray, rate: float) -> np.ndarray:
    """
    Randomly scale waveform amplitude (LightEQ augmentation)
    """
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data = data * np.random.uniform(1, 3)
    elif tmp < 2 * rate:
        data = data / np.random.uniform(1, 3)
    return data


def waveform_to_spectrogram(data: np.ndarray, fs: int = 100, nperseg: int = 80) -> np.ndarray:
    """
    Convert waveform to STFT spectrogram (LightEQ core preprocessing)
    
    Args:
        data: shape (time, channels) - typically (6000, 3)
        fs: sampling frequency (100 Hz for STEAD)
        nperseg: STFT window size
    
    Returns:
        spectrogram: shape (time_bins, freq_bins, channels) - typically (151, 41, 3)
    """
    n_channels = data.shape[1]
    
    # Compute STFT for first channel to get output shape
    f, t, Pxx = signal.stft(data[:, 0], fs=fs, nperseg=nperseg)
    spec_shape = (len(t), len(f), n_channels)
    
    spectrogram = np.empty(spec_shape, dtype=np.float32)
    
    for ch in range(n_channels):
        f, t, Pxx = signal.stft(data[:, ch], fs=fs, nperseg=nperseg)
        spectrogram[:, :, ch] = np.abs(Pxx).T
    
    return spectrogram


class STEADLightEQDataset(Dataset):
    """
    STEAD Dataset with LightEQ preprocessing pipeline
    
    Preprocessing:
    1. Load 3-channel waveform (6000 samples @ 100Hz)
    2. Optional augmentation (shift, noise, scale)
    3. Normalize (max or std)
    4. Convert to STFT spectrogram (151, 41, 3)
    
    Output shape: (3, 151, 41) for PyTorch (channels first)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        trace_list: List[str] = None,
        norm_mode: str = 'max',
        augmentation: bool = False,
        shift_event_r: float = 0.9,
        add_event_r: float = 0.5,
        add_noise_r: float = 0.4,
        scale_amplitude_r: float = None,
        fs: int = 100,
        nperseg: int = 80
    ):
        """
        Args:
            hdf5_path: Path to STEAD HDF5 file
            csv_path: Path to STEAD CSV metadata
            trace_list: List of trace names to use (if None, uses all)
            norm_mode: 'max' or 'std' normalization
            augmentation: Whether to apply data augmentation
            shift_event_r: Probability of shifting event
            add_event_r: Probability of adding second event
            add_noise_r: Probability of adding noise
            scale_amplitude_r: Probability of scaling amplitude
            fs: Sampling frequency (100 Hz for STEAD)
            nperseg: STFT window size
        """
        self.hdf5_path = hdf5_path
        self.norm_mode = norm_mode
        self.augmentation = augmentation
        self.shift_event_r = shift_event_r
        self.add_event_r = add_event_r
        self.add_noise_r = add_noise_r
        self.scale_amplitude_r = scale_amplitude_r
        self.fs = fs
        self.nperseg = nperseg
        
        # Load metadata
        print(f"Loading STEAD metadata from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if trace_list is not None:
            df = df[df['trace_name'].isin(trace_list)]
        
        self.trace_names = df['trace_name'].tolist()
        self.trace_categories = df['trace_category'].tolist()
        
        # Create labels: 0 = earthquake, 1 = noise
        self.labels = [0 if cat != 'noise' else 1 for cat in self.trace_categories]
        
        print(f"Loaded {len(self.trace_names)} traces")
        eq_count = sum(1 for l in self.labels if l == 0)
        noise_count = sum(1 for l in self.labels if l == 1)
        print(f"  Earthquakes: {eq_count}, Noise: {noise_count}")
        
    def __len__(self):
        return len(self.trace_names)
    
    def __getitem__(self, idx):
        trace_name = self.trace_names[idx]
        label = self.labels[idx]
        is_earthquake = (label == 0)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f.get(f'data/{trace_name}')
            if dataset is None:
                raise ValueError(f"Trace {trace_name} not found")
            
            data = np.array(dataset, dtype=np.float32)
            
            # Get metadata for augmentation
            snr = None
            spt = None
            sst = None
            coda_end = None
            
            if is_earthquake:
                try:
                    snr = np.array(dataset.attrs['snr_db'])
                    spt = int(dataset.attrs['p_arrival_sample'])
                    sst = int(dataset.attrs['s_arrival_sample'])
                    coda_end = int(dataset.attrs['coda_end_sample'])
                except:
                    pass
        
        # Apply augmentation (LightEQ style)
        if self.augmentation and is_earthquake and spt is not None:
            # Shift event
            if self.shift_event_r and snr is not None:
                data, spt, sst, coda_end = shift_event(
                    data, spt, sst, coda_end, snr, self.shift_event_r
                )
            
            # Add second event
            if self.add_event_r and snr is not None and coda_end is not None:
                data, _ = add_event(data, spt, sst, coda_end, snr, self.add_event_r)
            
            # Add noise
            if self.add_noise_r and snr is not None:
                data = add_noise(data, snr, self.add_noise_r)
            
            # Scale amplitude
            if self.scale_amplitude_r:
                data = scale_amplitude(data, self.scale_amplitude_r)
        
        # Normalize
        data = normalize(data, self.norm_mode)
        
        # Handle NaN values
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        
        # Convert to STFT spectrogram: (time, channels) -> (time_bins, freq_bins, channels)
        spectrogram = waveform_to_spectrogram(data, fs=self.fs, nperseg=self.nperseg)
        
        # Convert to PyTorch format: (channels, time_bins, freq_bins)
        spectrogram = np.transpose(spectrogram, (2, 0, 1))
        
        # Ensure contiguous array
        spectrogram = np.ascontiguousarray(spectrogram, dtype=np.float32)
        
        return torch.from_numpy(spectrogram), torch.tensor(label, dtype=torch.long)


def prepare_stead_splits(
    csv_path: str,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
    max_samples: int = None,
    balance_classes: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare train/val/test splits from STEAD dataset (LightEQ style)
    
    Args:
        csv_path: Path to STEAD CSV
        train_ratio: Ratio for training
        val_ratio: Ratio for validation
        test_ratio: Ratio for testing
        seed: Random seed
        max_samples: Maximum samples per class (for faster training)
        balance_classes: Whether to balance earthquake and noise classes
    
    Returns:
        train_traces, val_traces, test_traces
    """
    np.random.seed(seed)
    
    df = pd.read_csv(csv_path)
    
    # Separate earthquakes and noise
    eq_traces = df[df['trace_category'] != 'noise']['trace_name'].tolist()
    noise_traces = df[df['trace_category'] == 'noise']['trace_name'].tolist()
    
    print(f"Total earthquakes: {len(eq_traces)}")
    print(f"Total noise: {len(noise_traces)}")
    
    # Balance classes if requested
    if balance_classes:
        min_count = min(len(eq_traces), len(noise_traces))
        if max_samples:
            min_count = min(min_count, max_samples)
        
        np.random.shuffle(eq_traces)
        np.random.shuffle(noise_traces)
        eq_traces = eq_traces[:min_count]
        noise_traces = noise_traces[:min_count]
        print(f"Balanced to {min_count} samples per class")
    elif max_samples:
        np.random.shuffle(eq_traces)
        np.random.shuffle(noise_traces)
        eq_traces = eq_traces[:max_samples]
        noise_traces = noise_traces[:max_samples]
    
    # Combine and shuffle
    all_traces = eq_traces + noise_traces
    np.random.shuffle(all_traces)
    
    # Split
    n_total = len(all_traces)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_traces = all_traces[:n_train]
    val_traces = all_traces[n_train:n_train + n_val]
    test_traces = all_traces[n_train + n_val:]
    
    print(f"Train: {len(train_traces)}, Val: {len(val_traces)}, Test: {len(test_traces)}")
    
    return train_traces, val_traces, test_traces


def create_stead_lighteq_dataloaders(
    hdf5_path: str,
    csv_path: str,
    batch_size: int = 100,
    num_workers: int = 4,
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    max_samples: int = None,
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with LightEQ preprocessing
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Prepare splits
    train_traces, val_traces, test_traces = prepare_stead_splits(
        csv_path=csv_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_samples=max_samples
    )
    
    # Create datasets
    train_dataset = STEADLightEQDataset(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        trace_list=train_traces,
        augmentation=augmentation
    )
    
    val_dataset = STEADLightEQDataset(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        trace_list=val_traces,
        augmentation=False
    )
    
    test_dataset = STEADLightEQDataset(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        trace_list=test_traces,
        augmentation=False
    )
    
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


# For preprocessing DAS data with the same pipeline
class DASLightEQDataset(Dataset):
    """
    DAS Dataset with LightEQ-style STFT preprocessing
    
    Adapts DAS multi-channel data to work with LightEQ preprocessing
    """
    
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        window_size: int = 6000,
        stride: int = 3000,
        n_channels: int = 3,  # Use 3 channels like STEAD
        norm_mode: str = 'max',
        fs: int = 100,
        nperseg: int = 80
    ):
        """
        Args:
            file_paths: List of HDF5 file paths
            labels: List of labels (0=earthquake, 1=quarry blast)
            window_size: Samples per window (6000 for STEAD compatibility)
            stride: Stride between windows
            n_channels: Number of channels to extract (3 for STEAD compatibility)
            norm_mode: Normalization mode
            fs: Sampling frequency (resample DAS to 100Hz for STEAD compatibility)
            nperseg: STFT window size
        """
        self.file_paths = file_paths
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.n_channels = n_channels
        self.norm_mode = norm_mode
        self.fs = fs
        self.nperseg = nperseg
        
        # Build index of windows
        self.windows_index = []
        self._build_index()
    
    def _build_index(self):
        """Build index of all windows"""
        print("Building DAS window index...")
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Find data
                    data_key = None
                    for key in ['data', 'python_processing/sr/data', 'sr/data']:
                        if key in f or (key.count('/') > 0 and key.split('/')[0] in f):
                            try:
                                parts = key.split('/')
                                obj = f
                                for part in parts:
                                    obj = obj[part]
                                if hasattr(obj, 'shape'):
                                    data_key = key
                                    shape = obj.shape
                                    break
                            except:
                                continue
                    
                    if data_key is None:
                        print(f"Warning: Cannot find data in {file_path}")
                        continue
                    
                    # Determine orientation
                    if shape[0] > shape[1]:
                        num_samples, num_channels = shape
                        is_transposed = False
                    else:
                        num_channels, num_samples = shape
                        is_transposed = True
                    
                    # Calculate windows
                    num_windows = max(1, (num_samples - self.window_size) // self.stride + 1)
                    
                    for win_idx in range(num_windows):
                        self.windows_index.append({
                            'file_idx': file_idx,
                            'start_idx': win_idx * self.stride,
                            'data_key': data_key,
                            'is_transposed': is_transposed,
                            'num_channels': num_channels
                        })
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Total DAS windows: {len(self.windows_index)}")
    
    def __len__(self):
        return len(self.windows_index)
    
    def __getitem__(self, idx):
        info = self.windows_index[idx]
        file_path = self.file_paths[info['file_idx']]
        label = self.labels[info['file_idx']]
        
        with h5py.File(file_path, 'r') as f:
            # Navigate to dataset
            parts = info['data_key'].split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            
            start = info['start_idx']
            end = start + self.window_size
            
            if info['is_transposed']:
                # (channels, time)
                data = obj[:, start:end]
                data = np.array(data, dtype=np.float32).T  # -> (time, channels)
            else:
                # (time, channels)
                data = obj[start:end, :]
                data = np.array(data, dtype=np.float32)
            
            # Pad if needed
            if data.shape[0] < self.window_size:
                pad = np.zeros((self.window_size - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, pad])
        
        # Extract n_channels evenly spaced across DAS array
        total_channels = data.shape[1]
        if total_channels > self.n_channels:
            # Select evenly spaced channels (like E, N, Z components)
            indices = np.linspace(0, total_channels - 1, self.n_channels, dtype=int)
            data = data[:, indices]
        elif total_channels < self.n_channels:
            # Pad with zeros
            pad = np.zeros((data.shape[0], self.n_channels - total_channels), dtype=np.float32)
            data = np.hstack([data, pad])
        
        # Resample to 100Hz if DAS is 200Hz (STEAD compatibility)
        # Simple decimation by factor of 2
        if data.shape[0] == self.window_size and self.window_size > 6000:
            # Downsample
            from scipy.signal import decimate
            data = decimate(data, 2, axis=0)
        
        # Ensure correct length
        if data.shape[0] > 6000:
            data = data[:6000, :]
        elif data.shape[0] < 6000:
            pad = np.zeros((6000 - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, pad])
        
        # Normalize
        data = normalize(data, self.norm_mode)
        
        # Handle NaN
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        
        # Convert to spectrogram
        spectrogram = waveform_to_spectrogram(data, fs=self.fs, nperseg=self.nperseg)
        
        # PyTorch format: (channels, time, freq)
        spectrogram = np.transpose(spectrogram, (2, 0, 1))
        spectrogram = np.ascontiguousarray(spectrogram, dtype=np.float32)
        
        return torch.from_numpy(spectrogram), torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    # Test the dataset
    print("Testing STEADLightEQDataset...")
    
    # Check if STEAD exists
    stead_hdf5 = Path('data/stead/merge.hdf5')
    stead_csv = Path('data/stead/merge.csv')
    
    if stead_hdf5.exists() and stead_csv.exists():
        train_loader, val_loader, test_loader = create_stead_lighteq_dataloaders(
            hdf5_path=str(stead_hdf5),
            csv_path=str(stead_csv),
            batch_size=32,
            max_samples=1000  # Small sample for testing
        )
        
        # Get a batch
        for X, y in train_loader:
            print(f"Batch shape: {X.shape}")  # Should be (batch, 3, 151, 41)
            print(f"Labels shape: {y.shape}")
            print(f"Label distribution: {torch.bincount(y)}")
            break
    else:
        print("STEAD dataset not found. Please download it first.")
        print("Run: python setup_stead.py")
