"""
Advanced signal processing for DAS seismic data:
- Intelligent chunking around seismic events
- Noise filtering (90% reduction)
- P and S wave arrival time picking
"""

import h5py
import numpy as np
from scipy import signal, ndimage
from scipy.signal import butter, filtfilt, hilbert
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class SeismicSignalProcessor:
    """
    Process DAS seismic data to extract event windows and pick arrivals
    """
    
    def __init__(
        self,
        sample_rate: float = 200.0,  # Hz
        p_freq_band: Tuple[float, float] = (5, 50),  # Hz, P-wave frequency
        s_freq_band: Tuple[float, float] = (2, 30),  # Hz, S-wave frequency
        noise_freq_band: Tuple[float, float] = (0.5, 100),  # Hz, for noise estimation
    ):
        """
        Args:
            sample_rate: Sampling rate in Hz
            p_freq_band: Frequency band for P-wave detection (min, max)
            s_freq_band: Frequency band for S-wave detection (min, max)
            noise_freq_band: Frequency band for noise estimation
        """
        self.sample_rate = sample_rate
        self.p_freq_band = p_freq_band
        self.s_freq_band = s_freq_band
        self.noise_freq_band = noise_freq_band
        
    def bandpass_filter(self, data: np.ndarray, freq_band: Tuple[float, float]) -> np.ndarray:
        """
        Apply bandpass Butterworth filter
        
        Args:
            data: Input array (channels, time) or (time,)
            freq_band: (low_freq, high_freq) in Hz
            
        Returns:
            Filtered data
        """
        nyquist = self.sample_rate / 2
        low = freq_band[0] / nyquist
        high = freq_band[1] / nyquist
        
        # Ensure frequencies are valid
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            return data
        
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter along time axis
        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            return filtfilt(b, a, data, axis=1)
    
    def compute_sta_lta(
        self,
        data: np.ndarray,
        sta_window: float = 0.5,  # seconds
        lta_window: float = 5.0,   # seconds
    ) -> np.ndarray:
        """
        Compute STA/LTA (Short-Term Average / Long-Term Average) ratio
        Classic earthquake detection algorithm
        
        Args:
            data: Input trace (time,) or (channels, time)
            sta_window: Short-term window in seconds
            lta_window: Long-term window in seconds
            
        Returns:
            STA/LTA ratio
        """
        sta_samples = int(sta_window * self.sample_rate)
        lta_samples = int(lta_window * self.sample_rate)
        
        # Square the signal for energy
        energy = data ** 2
        
        if data.ndim == 1:
            # 1D case
            sta = ndimage.uniform_filter1d(energy, sta_samples, mode='constant')
            lta = ndimage.uniform_filter1d(energy, lta_samples, mode='constant')
        else:
            # 2D case - apply along time axis
            sta = ndimage.uniform_filter1d(energy, sta_samples, axis=1, mode='constant')
            lta = ndimage.uniform_filter1d(energy, lta_samples, axis=1, mode='constant')
        
        # Avoid division by zero
        sta_lta = np.divide(sta, lta, where=lta > 0, out=np.zeros_like(sta))
        
        return sta_lta
    
    def estimate_noise_level(self, data: np.ndarray, pre_event_samples: int = 5000) -> float:
        """
        Estimate noise level from pre-event window
        
        Args:
            data: Input data (channels, time)
            pre_event_samples: Number of samples to use for noise estimation
            
        Returns:
            Noise standard deviation
        """
        # Use first N samples as noise window
        noise_window = data[:, :min(pre_event_samples, data.shape[1] // 4)]
        
        # Filter to target frequency band
        filtered_noise = self.bandpass_filter(noise_window, self.noise_freq_band)
        
        # Compute robust noise estimate (MAD estimator)
        mad = np.median(np.abs(filtered_noise - np.median(filtered_noise)))
        noise_std = 1.4826 * mad  # Convert MAD to standard deviation
        
        return noise_std
    
    def detect_event_window(
        self,
        data: np.ndarray,
        noise_threshold: float = 3.0,  # SNR threshold
        sta_lta_threshold: float = 2.5,
        min_event_duration: float = 2.0,  # seconds
        pre_event_buffer: float = 5.0,  # seconds before trigger
        post_event_buffer: float = 15.0,  # seconds after trigger
    ) -> Optional[Tuple[int, int]]:
        """
        Detect seismic event window using energy and STA/LTA
        
        Args:
            data: Input data (channels, time)
            noise_threshold: SNR threshold for detection
            sta_lta_threshold: STA/LTA threshold
            min_event_duration: Minimum event duration in seconds
            pre_event_buffer: Pre-event buffer in seconds
            post_event_buffer: Post-event buffer in seconds
            
        Returns:
            (start_sample, end_sample) or None if no event detected
        """
        # Compute energy trace (sum over channels)
        energy_trace = np.sum(data ** 2, axis=0)
        
        # Filter for event detection
        filtered_energy = self.bandpass_filter(energy_trace, self.noise_freq_band)
        
        # Compute STA/LTA
        sta_lta = self.compute_sta_lta(filtered_energy, sta_window=0.5, lta_window=5.0)
        
        # Find trigger point (first crossing of threshold)
        trigger_mask = sta_lta > sta_lta_threshold
        trigger_indices = np.where(trigger_mask)[0]
        
        if len(trigger_indices) == 0:
            return None
        
        trigger_idx = trigger_indices[0]
        
        # Find end of event (when STA/LTA drops below threshold)
        end_mask = sta_lta[trigger_idx:] < sta_lta_threshold * 0.5
        end_indices = np.where(end_mask)[0]
        
        if len(end_indices) > 0:
            end_idx = trigger_idx + end_indices[0]
        else:
            end_idx = len(sta_lta) - 1
        
        # Check minimum duration
        event_duration = (end_idx - trigger_idx) / self.sample_rate
        if event_duration < min_event_duration:
            # Extend to minimum duration
            end_idx = int(trigger_idx + min_event_duration * self.sample_rate)
            end_idx = min(end_idx, len(sta_lta) - 1)
        
        # Add buffers
        start_sample = max(0, int(trigger_idx - pre_event_buffer * self.sample_rate))
        end_sample = min(len(sta_lta) - 1, int(end_idx + post_event_buffer * self.sample_rate))
        
        return (start_sample, end_sample)
    
    def pick_p_wave(
        self,
        data: np.ndarray,
        search_window: Optional[Tuple[int, int]] = None,
        threshold_factor: float = 3.0,
    ) -> Dict[str, any]:
        """
        Pick P-wave arrival time using characteristic function
        
        Args:
            data: Input data (channels, time)
            search_window: (start_sample, end_sample) to search in
            threshold_factor: Threshold as multiple of noise level
            
        Returns:
            Dictionary with P-wave pick information
        """
        if search_window is None:
            search_window = (0, data.shape[1])
        
        start_idx, end_idx = search_window
        window_data = data[:, start_idx:end_idx]
        
        # Filter in P-wave frequency band
        filtered = self.bandpass_filter(window_data, self.p_freq_band)
        
        # Compute characteristic function (envelope)
        cf = np.sum(np.abs(hilbert(filtered, axis=1)), axis=0)
        
        # Smooth CF
        cf_smooth = ndimage.gaussian_filter1d(cf, sigma=self.sample_rate * 0.1)
        
        # Estimate noise level
        noise_std = np.std(cf_smooth[:int(len(cf_smooth) * 0.2)])
        threshold = noise_std * threshold_factor
        
        # Find first crossing
        crossings = np.where(cf_smooth > threshold)[0]
        
        if len(crossings) == 0:
            return {
                'pick_sample': None,
                'pick_time': None,
                'confidence': 0.0,
                'snr': 0.0
            }
        
        pick_idx = crossings[0]
        pick_sample = start_idx + pick_idx
        pick_time = pick_sample / self.sample_rate
        
        # Compute SNR
        signal_level = np.max(cf_smooth[pick_idx:pick_idx + int(self.sample_rate)])
        snr = signal_level / noise_std if noise_std > 0 else 0.0
        
        # Compute confidence (0-1)
        confidence = min(1.0, snr / 10.0)
        
        return {
            'pick_sample': pick_sample,
            'pick_time': pick_time,
            'confidence': confidence,
            'snr': snr,
            'characteristic_function': cf_smooth
        }
    
    def pick_s_wave(
        self,
        data: np.ndarray,
        p_pick_sample: int,
        search_window_duration: float = 10.0,  # seconds after P
        threshold_factor: float = 2.5,
    ) -> Dict[str, any]:
        """
        Pick S-wave arrival time (after P-wave)
        
        Args:
            data: Input data (channels, time)
            p_pick_sample: P-wave pick sample index
            search_window_duration: How long to search after P-wave
            threshold_factor: Threshold as multiple of noise level
            
        Returns:
            Dictionary with S-wave pick information
        """
        # Define search window starting after P-wave
        start_idx = p_pick_sample + int(0.5 * self.sample_rate)  # Start 0.5s after P
        end_idx = min(
            data.shape[1],
            int(p_pick_sample + search_window_duration * self.sample_rate)
        )
        
        if start_idx >= end_idx:
            return {
                'pick_sample': None,
                'pick_time': None,
                'confidence': 0.0,
                'snr': 0.0
            }
        
        window_data = data[:, start_idx:end_idx]
        
        # Filter in S-wave frequency band (lower than P)
        filtered = self.bandpass_filter(window_data, self.s_freq_band)
        
        # Compute characteristic function using polarization change
        # Use horizontal energy increase
        cf = np.sum(filtered ** 2, axis=0)
        
        # Smooth CF
        cf_smooth = ndimage.gaussian_filter1d(cf, sigma=self.sample_rate * 0.15)
        
        # Estimate noise level from early part
        noise_std = np.std(cf_smooth[:int(len(cf_smooth) * 0.2)])
        threshold = noise_std * threshold_factor
        
        # Find significant increase in energy
        crossings = np.where(cf_smooth > threshold)[0]
        
        if len(crossings) == 0:
            return {
                'pick_sample': None,
                'pick_time': None,
                'confidence': 0.0,
                'snr': 0.0
            }
        
        pick_idx = crossings[0]
        pick_sample = start_idx + pick_idx
        pick_time = pick_sample / self.sample_rate
        
        # Compute SNR
        signal_level = np.max(cf_smooth[pick_idx:pick_idx + int(self.sample_rate)])
        snr = signal_level / noise_std if noise_std > 0 else 0.0
        
        # Compute confidence
        confidence = min(1.0, snr / 8.0)
        
        return {
            'pick_sample': pick_sample,
            'pick_time': pick_time,
            'confidence': confidence,
            'snr': snr,
            'characteristic_function': cf_smooth
        }
    
    def filter_noise_spatial(
        self,
        data: np.ndarray,
        noise_reduction: float = 0.9,  # Target 90% reduction
        method: str = 'fk'  # 'fk' or 'median'
    ) -> np.ndarray:
        """
        Apply spatial noise filtering to reduce noise by target amount
        
        Args:
            data: Input data (channels, time)
            noise_reduction: Target noise reduction fraction (0-1)
            method: Filtering method ('fk' for FK filter, 'median' for median filter)
            
        Returns:
            Filtered data
        """
        if method == 'median':
            # Median filter in spatial dimension (good for coherent noise)
            kernel_size = max(3, int(data.shape[0] * 0.01))  # 1% of channels
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            filtered = ndimage.median_filter(data, size=(kernel_size, 1))
            
            # Blend based on noise reduction target
            alpha = 1.0 - noise_reduction
            result = alpha * data + (1 - alpha) * filtered
            
        elif method == 'fk':
            # FK (frequency-wavenumber) filtering
            # This is more sophisticated for DAS data
            result = self._fk_filter(data, noise_reduction)
        else:
            result = data
        
        return result
    
    def _fk_filter(self, data: np.ndarray, noise_reduction: float) -> np.ndarray:
        """
        Apply FK filtering to suppress low-velocity noise
        """
        # 2D FFT
        fk_spectrum = np.fft.fft2(data)
        fk_spectrum_shifted = np.fft.fftshift(fk_spectrum)
        
        # Create velocity filter (reject low apparent velocities)
        ny, nx = fk_spectrum_shifted.shape
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        
        # Apparent velocity threshold (m/s)
        v_threshold = 1000  # Reject signals slower than 1000 m/s
        channel_spacing = 10  # meters (typical DAS)
        
        # Create cone filter in FK domain
        with np.errstate(divide='ignore', invalid='ignore'):
            apparent_v = np.abs(KX * self.sample_rate / (KY * channel_spacing + 1e-10))
            filter_mask = apparent_v > v_threshold
        
        # Apply filter with scaling based on noise reduction
        filter_strength = noise_reduction
        fk_filtered = fk_spectrum_shifted * (1 - filter_strength * (1 - filter_mask.astype(float)))
        
        # Inverse FFT
        fk_filtered_unshifted = np.fft.ifftshift(fk_filtered)
        filtered_data = np.real(np.fft.ifft2(fk_filtered_unshifted))
        
        return filtered_data
    
    def process_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = False,
        decimate_channels: int = 10,  # Use every Nth channel for speed
    ) -> Dict[str, any]:
        """
        Process a single DAS file: detect event, filter noise, pick P and S waves
        
        Args:
            file_path: Path to HDF5 file
            output_dir: Directory to save processed chunks
            visualize: Whether to create visualization plots
            decimate_channels: Use every Nth channel (1=all, 10=every 10th)
            
        Returns:
            Dictionary with processing results
        """
        print(f"\nProcessing: {Path(file_path).name}")
        
        # Load data
        with h5py.File(file_path, 'r') as f:
            # Try to find strain rate data - check multiple possible paths
            data = None
            time = None
            distance = None
            
            # Try different possible paths
            possible_paths = [
                'python_processing/sr',  # DAS-BIGORRE format
                'sr',                     # Alternative format
                'data',                   # Simple format
            ]
            
            for path in possible_paths:
                try:
                    if path in f or path.split('/')[0] in f:
                        # Construct full paths
                        if path == 'python_processing/sr':
                            data = f['python_processing/sr/data'][::decimate_channels, :]
                            time = f['python_processing/sr/time'][:]
                            distance = f['python_processing/sr/distance'][::decimate_channels]
                        elif path == 'sr':
                            data = f['sr/data'][::decimate_channels, :]
                            time = f['sr/time'][:]
                            distance = f['sr/distance'][::decimate_channels]
                        elif 'data' in f:
                            data = f['data'][::decimate_channels, :]
                            time = np.arange(data.shape[1]) / self.sample_rate
                            distance = np.arange(data.shape[0]) * 10
                        break
                except (KeyError, AttributeError):
                    continue
            
            if data is None:
                print(f"Could not find data in {file_path}")
                print(f"Available keys: {list(f.keys())}")
                return None
        
        print(f"  Data shape: {data.shape}")
        print(f"  Duration: {data.shape[1] / self.sample_rate:.1f} seconds")
        
        # Step 1: Filter noise (90% reduction)
        print("  Filtering noise...")
        filtered_data = self.filter_noise_spatial(data, noise_reduction=0.9, method='median')
        noise_reduction_actual = 1.0 - (np.std(filtered_data) / np.std(data))
        print(f"  Noise reduction: {noise_reduction_actual * 100:.1f}%")
        
        # Step 2: Detect event window
        print("  Detecting event window...")
        event_window = self.detect_event_window(filtered_data)
        
        if event_window is None:
            print("  Warning: No event detected!")
            event_window = (0, filtered_data.shape[1])
        else:
            start_time = event_window[0] / self.sample_rate
            end_time = event_window[1] / self.sample_rate
            print(f"  Event window: {start_time:.1f}s - {end_time:.1f}s")
        
        # Step 3: Pick P-wave
        print("  Picking P-wave...")
        p_pick = self.pick_p_wave(filtered_data, search_window=event_window)
        
        if p_pick['pick_sample'] is not None:
            print(f"  P-wave pick: {p_pick['pick_time']:.2f}s (SNR: {p_pick['snr']:.1f}, confidence: {p_pick['confidence']:.2f})")
        else:
            print("  P-wave pick: Not detected")
        
        # Step 4: Pick S-wave (if P was detected)
        s_pick = None
        if p_pick['pick_sample'] is not None:
            print("  Picking S-wave...")
            s_pick = self.pick_s_wave(filtered_data, p_pick['pick_sample'])
            
            if s_pick['pick_sample'] is not None:
                print(f"  S-wave pick: {s_pick['pick_time']:.2f}s (SNR: {s_pick['snr']:.1f}, confidence: {s_pick['confidence']:.2f})")
                p_s_time = s_pick['pick_time'] - p_pick['pick_time']
                print(f"  P-S time difference: {p_s_time:.2f}s")
            else:
                print("  S-wave pick: Not detected")
        
        # Step 5: Extract and save chunk
        chunk_data = filtered_data[:, event_window[0]:event_window[1]]
        
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = Path(file_path).stem + "_processed.h5"
            output_file = output_path / filename
            
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('data', data=chunk_data, compression='gzip')
                f.create_dataset('time', data=time[event_window[0]:event_window[1]])
                f.create_dataset('distance', data=distance)
                
                # Store picks as attributes
                if p_pick['pick_sample'] is not None:
                    f.attrs['p_pick_time'] = p_pick['pick_time']
                    f.attrs['p_pick_confidence'] = p_pick['confidence']
                    f.attrs['p_pick_snr'] = p_pick['snr']
                
                if s_pick is not None and s_pick['pick_sample'] is not None:
                    f.attrs['s_pick_time'] = s_pick['pick_time']
                    f.attrs['s_pick_confidence'] = s_pick['confidence']
                    f.attrs['s_pick_snr'] = s_pick['snr']
            
            print(f"  Saved to: {output_file}")
        
        # Step 6: Visualize
        if visualize:
            self._visualize_results(
                data, filtered_data, chunk_data, event_window,
                p_pick, s_pick, time, distance, file_path
            )
        
        return {
            'file': file_path,
            'original_shape': data.shape,
            'chunk_shape': chunk_data.shape,
            'event_window': event_window,
            'noise_reduction': noise_reduction_actual,
            'p_pick': p_pick,
            's_pick': s_pick,
        }
    
    def _visualize_results(
        self,
        original_data: np.ndarray,
        filtered_data: np.ndarray,
        chunk_data: np.ndarray,
        event_window: Tuple[int, int],
        p_pick: Dict,
        s_pick: Optional[Dict],
        time: np.ndarray,
        distance: np.ndarray,
        file_path: str,
    ):
        """Create visualization of processing results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Compute display range
        vmax = np.percentile(np.abs(original_data), 99)
        
        # Original data
        im0 = axes[0, 0].imshow(
            original_data, aspect='auto', cmap='seismic',
            vmin=-vmax, vmax=vmax, extent=[0, time[-1], distance[-1]/1000, distance[0]/1000]
        )
        axes[0, 0].set_title('Original Data')
        axes[0, 0].set_ylabel('Distance (km)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Filtered data
        im1 = axes[1, 0].imshow(
            filtered_data, aspect='auto', cmap='seismic',
            vmin=-vmax, vmax=vmax, extent=[0, time[-1], distance[-1]/1000, distance[0]/1000]
        )
        axes[1, 0].set_title('Filtered Data (90% noise reduction)')
        axes[1, 0].set_ylabel('Distance (km)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Mark event window
        axes[1, 0].axvline(event_window[0] / self.sample_rate, color='yellow', linestyle='--', linewidth=2)
        axes[1, 0].axvline(event_window[1] / self.sample_rate, color='yellow', linestyle='--', linewidth=2)
        
        # Chunk data
        chunk_time = time[event_window[0]:event_window[1]]
        im2 = axes[2, 0].imshow(
            chunk_data, aspect='auto', cmap='seismic',
            vmin=-vmax, vmax=vmax, extent=[chunk_time[0], chunk_time[-1], distance[-1]/1000, distance[0]/1000]
        )
        axes[2, 0].set_title('Extracted Event Chunk')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Distance (km)')
        plt.colorbar(im2, ax=axes[2, 0])
        
        # Mark P and S picks
        if p_pick['pick_sample'] is not None:
            axes[2, 0].axvline(p_pick['pick_time'], color='red', linestyle='-', linewidth=2, label='P-wave')
        if s_pick is not None and s_pick['pick_sample'] is not None:
            axes[2, 0].axvline(s_pick['pick_time'], color='blue', linestyle='-', linewidth=2, label='S-wave')
        axes[2, 0].legend()
        
        # Energy trace with STA/LTA
        energy = np.sum(filtered_data ** 2, axis=0)
        sta_lta = self.compute_sta_lta(energy)
        
        axes[0, 1].plot(time, energy / np.max(energy), linewidth=0.5, label='Energy')
        axes[0, 1].set_ylabel('Normalized Energy')
        axes[0, 1].set_title('Energy and STA/LTA')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        ax_twin = axes[0, 1].twinx()
        ax_twin.plot(time, sta_lta, 'r', linewidth=0.5, label='STA/LTA')
        ax_twin.set_ylabel('STA/LTA', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax_twin.legend(loc='upper right')
        
        # P-wave characteristic function
        if p_pick['pick_sample'] is not None and 'characteristic_function' in p_pick:
            cf_time = np.linspace(0, time[-1], len(p_pick['characteristic_function']))
            axes[1, 1].plot(cf_time, p_pick['characteristic_function'], linewidth=1)
            axes[1, 1].axvline(p_pick['pick_time'], color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_title(f"P-wave CF (SNR: {p_pick['snr']:.1f}, Conf: {p_pick['confidence']:.2f})")
            axes[1, 1].set_ylabel('Amplitude')
            axes[1, 1].grid(True, alpha=0.3)
        
        # S-wave characteristic function
        if s_pick is not None and s_pick['pick_sample'] is not None and 'characteristic_function' in s_pick:
            # Reconstruct time for S-wave CF
            search_start = p_pick['pick_sample'] + int(0.5 * self.sample_rate)
            cf_time = np.linspace(
                search_start / self.sample_rate,
                min(time[-1], search_start / self.sample_rate + 10),
                len(s_pick['characteristic_function'])
            )
            axes[2, 1].plot(cf_time, s_pick['characteristic_function'], linewidth=1)
            axes[2, 1].axvline(s_pick['pick_time'], color='blue', linestyle='--', linewidth=2)
            axes[2, 1].set_title(f"S-wave CF (SNR: {s_pick['snr']:.1f}, Conf: {s_pick['confidence']:.2f})")
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Amplitude')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle(Path(file_path).name, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = Path('visualizations')
        output_path.mkdir(exist_ok=True)
        fig_path = output_path / (Path(file_path).stem + '_processing.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: {fig_path}")
        plt.close()


def process_all_files(
    data_dir: str = 'data',
    output_dir: str = 'data/processed_chunks',
    visualize: bool = True,
    max_files: Optional[int] = None,
    save_picks_summary: bool = True,
):
    """
    Process all DAS files in directory with memory-efficient handling
    
    Args:
        data_dir: Directory containing HDF5 files
        output_dir: Directory to save processed chunks
        visualize: Whether to create visualizations
        max_files: Maximum number of files to process (None = all)
        save_picks_summary: Save CSV with all picks
    """
    import gc
    import csv
    from datetime import datetime
    
    processor = SeismicSignalProcessor(sample_rate=200.0)
    
    # Find all HDF5 files (excluding processed ones and subdirectories)
    data_path = Path(data_dir)
    h5_files = sorted([f for f in data_path.glob('DAS-BIGORRE_*.h5') 
                       if f.is_file() and 'processed' not in f.name])
    
    if max_files is not None:
        h5_files = h5_files[:max_files]
    
    print("="*70)
    print(f"PROCESSING ALL DAS FILES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"Found {len(h5_files)} files to process")
    print(f"Output directory: {output_dir}")
    print(f"Noise reduction target: 90%")
    print("="*70)
    
    results = []
    picks_data = []
    
    for i, file_path in enumerate(h5_files, 1):
        print(f"\n[{i}/{len(h5_files)}] Processing: {file_path.name}")
        print("-" * 70)
        
        try:
            # Process file
            result = processor.process_file(
                str(file_path),
                output_dir=output_dir,
                visualize=visualize,
                decimate_channels=10  # Use every 10th channel for memory efficiency
            )
            
            if result is not None:
                results.append(result)
                
                # Collect picks for summary
                pick_info = {
                    'filename': file_path.name,
                    'p_pick_time': result['p_pick']['pick_time'] if result['p_pick']['pick_sample'] is not None else None,
                    'p_pick_confidence': result['p_pick']['confidence'] if result['p_pick']['pick_sample'] is not None else 0.0,
                    'p_pick_snr': result['p_pick']['snr'] if result['p_pick']['pick_sample'] is not None else 0.0,
                    's_pick_time': result['s_pick']['pick_time'] if result['s_pick'] and result['s_pick']['pick_sample'] is not None else None,
                    's_pick_confidence': result['s_pick']['confidence'] if result['s_pick'] and result['s_pick']['pick_sample'] is not None else 0.0,
                    's_pick_snr': result['s_pick']['snr'] if result['s_pick'] and result['s_pick']['pick_sample'] is not None else 0.0,
                    'p_s_diff': (result['s_pick']['pick_time'] - result['p_pick']['pick_time']) 
                                if result['s_pick'] and result['s_pick']['pick_sample'] is not None 
                                and result['p_pick']['pick_sample'] is not None else None,
                    'noise_reduction_percent': result['noise_reduction'] * 100,
                    'event_start_time': result['event_window'][0] / 200.0,
                    'event_end_time': result['event_window'][1] / 200.0,
                }
                picks_data.append(pick_info)
                
                print(f"✓ Completed: {file_path.name}")
            else:
                print(f"✗ Failed: {file_path.name}")
                
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            # Explicit memory cleanup after each file
            gc.collect()
            print(f"Memory freed for next file")
    
    # Save picks summary to CSV
    if save_picks_summary and picks_data:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / 'picks_summary.csv'
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'p_pick_time', 'p_pick_confidence', 'p_pick_snr',
                         's_pick_time', 's_pick_confidence', 's_pick_snr', 'p_s_diff',
                         'noise_reduction_percent', 'event_start_time', 'event_end_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for pick in picks_data:
                writer.writerow(pick)
        
        print(f"\n✓ Picks summary saved to: {csv_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(results)}/{len(h5_files)}")
    
    if results:
        total_p_picks = sum(1 for r in results if r['p_pick']['pick_sample'] is not None)
        total_s_picks = sum(1 for r in results if r['s_pick'] and r['s_pick']['pick_sample'] is not None)
        avg_noise_reduction = np.mean([r['noise_reduction'] for r in results]) * 100
        avg_time_compression = np.mean([1 - r['chunk_shape'][1]/r['original_shape'][1] for r in results]) * 100
        
        print(f"\nP-wave picks detected: {total_p_picks}/{len(results)} ({total_p_picks/len(results)*100:.1f}%)")
        print(f"S-wave picks detected: {total_s_picks}/{len(results)} ({total_s_picks/len(results)*100:.1f}%)")
        print(f"Average noise reduction: {avg_noise_reduction:.1f}%")
        print(f"Average time compression: {avg_time_compression:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 70)
        for result in results:
            filename = Path(result['file']).name
            print(f"\n{filename}")
            print(f"  Original: {result['original_shape']}, Chunk: {result['chunk_shape']}")
            print(f"  Noise reduction: {result['noise_reduction']*100:.1f}%")
            
            if result['p_pick']['pick_sample'] is not None:
                print(f"  P-pick: {result['p_pick']['pick_time']:.2f}s (SNR: {result['p_pick']['snr']:.1f}, conf: {result['p_pick']['confidence']:.2f})")
            else:
                print(f"  P-pick: NOT DETECTED")
            
            if result['s_pick'] is not None and result['s_pick']['pick_sample'] is not None:
                print(f"  S-pick: {result['s_pick']['pick_time']:.2f}s (SNR: {result['s_pick']['snr']:.1f}, conf: {result['s_pick']['confidence']:.2f})")
                p_s_diff = result['s_pick']['pick_time'] - result['p_pick']['pick_time']
                print(f"  P-S difference: {p_s_diff:.2f}s")
            else:
                print(f"  S-pick: NOT DETECTED")
    
    print("\n" + "="*70)
    print(f"Processing completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    visualize = '--no-viz' not in sys.argv
    max_files = None
    
    # Check for max_files argument
    for arg in sys.argv[1:]:
        if arg.startswith('--max='):
            max_files = int(arg.split('=')[1])
    
    print("\n" + "="*70)
    print("DAS SEISMIC DATA PROCESSING")
    print("="*70)
    print("Features:")
    print("  • 90% noise filtering (spatial median filter)")
    print("  • Automatic event detection (STA/LTA)")
    print("  • P-wave picking (5-50 Hz)")
    print("  • S-wave picking (2-30 Hz)")
    print("  • Memory-efficient processing (clears RAM after each file)")
    print("="*70)
    
    # Process all files
    results = process_all_files(
        data_dir='data',
        output_dir='data/processed_chunks',
        visualize=visualize,
        max_files=max_files,
        save_picks_summary=True
    )
    
    print("\n✓ All processing complete!")
    print(f"  Processed files: data/processed_chunks/")
    print(f"  Picks summary: data/processed_chunks/picks_summary.csv")
    if visualize:
        print(f"  Visualizations: visualizations/")
    print("\nUsage:")
    print("  python signal_processing.py              # Process all files with visualization")
    print("  python signal_processing.py --no-viz     # Process without visualization")
    print("  python signal_processing.py --max=5      # Process first 5 files only")
