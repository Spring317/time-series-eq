"""
Inference script for trained seismic classifier
Load a trained model and classify new DAS data
"""

import torch
import h5py
import numpy as np
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from models import create_model


class SeismicClassifier:
    """
    Inference wrapper for seismic classification
    """
    
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to config file (optional, loaded from checkpoint)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint['config']
        
        # Get number of channels from checkpoint
        model_state = checkpoint['model_state_dict']
        
        # Infer num_channels from first layer
        first_key = list(model_state.keys())[0]
        if 'channel_proj' in first_key or 'input_proj' in first_key:
            first_param = model_state[first_key]
            num_channels = first_param.shape[1]
        else:
            num_channels = self.config['data']['num_channels']
        
        # Build model
        self.model = create_model(self.config, num_channels)
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.window_size = self.config['data']['window_size']
        self.stride = self.config['data']['stride']
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_data(self, data):
        """
        Preprocess raw data
        Args:
            data: numpy array of shape (time, channels) or (channels, time)
        Returns:
            preprocessed data of shape (channels, time)
        """
        # Ensure correct orientation
        if data.shape[0] > data.shape[1]:
            data = data.T  # (time, channels) -> (channels, time)
        
        # Normalize per channel
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        data = (data - mean) / std
        
        return data.astype(np.float32)
    
    def predict_window(self, data):
        """
        Predict a single window
        Args:
            data: numpy array of shape (channels, time)
        Returns:
            class_idx, probabilities
        """
        # Convert to tensor
        data_tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(data_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        
        return pred_class.item(), probs.cpu().numpy()[0]
    
    def predict_file(self, file_path, aggregate='vote'):
        """
        Predict class for an entire file using sliding window
        Args:
            file_path: Path to HDF5 file
            aggregate: How to aggregate predictions ('vote', 'average')
        Returns:
            predicted_class, confidence
        """
        # Load data
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            if len(keys) == 0:
                raise ValueError(f"No data in {file_path}")
            
            data_key = keys[0]
            raw_data = f[data_key][:]
        
        # Preprocess
        data = self.preprocess_data(raw_data)
        num_channels, num_samples = data.shape
        
        # Slide window over data
        predictions = []
        probabilities = []
        
        num_windows = max(1, (num_samples - self.window_size) // self.stride + 1)
        
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            window = data[:, start_idx:end_idx]
            
            # Pad if necessary
            if window.shape[1] < self.window_size:
                pad_width = ((0, 0), (0, self.window_size - window.shape[1]))
                window = np.pad(window, pad_width, mode='constant')
            
            pred_class, probs = self.predict_window(window)
            predictions.append(pred_class)
            probabilities.append(probs)
        
        # Aggregate predictions
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        if aggregate == 'vote':
            # Majority vote
            final_class = np.bincount(predictions).argmax()
            confidence = np.mean(probabilities[:, final_class])
        else:  # average
            # Average probabilities
            avg_probs = np.mean(probabilities, axis=0)
            final_class = np.argmax(avg_probs)
            confidence = avg_probs[final_class]
        
        return final_class, confidence
    
    def predict_batch(self, file_paths):
        """
        Predict classes for multiple files
        Args:
            file_paths: List of paths to HDF5 files
        Returns:
            List of (file_path, predicted_class, confidence) tuples
        """
        results = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                pred_class, confidence = self.predict_file(file_path)
                results.append((file_path, pred_class, confidence))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append((file_path, -1, 0.0))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Seismic event classification inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input HDF5 file or directory')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Path to output JSON file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--aggregate', type=str, default='vote',
                        choices=['vote', 'average'],
                        help='Method to aggregate window predictions')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = SeismicClassifier(
        args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        file_paths = [str(input_path)]
    elif input_path.is_dir():
        file_paths = [str(p) for p in sorted(input_path.glob('*.h5'))]
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print(f"\nProcessing {len(file_paths)} files...")
    
    # Run predictions
    results = classifier.predict_batch(file_paths)
    
    # Format results
    class_names = ['Earthquake', 'Quarry Blast']
    formatted_results = []
    
    print("\nResults:")
    print("-" * 80)
    for file_path, pred_class, confidence in results:
        if pred_class >= 0:
            class_name = class_names[pred_class]
            print(f"{Path(file_path).name:50s} {class_name:15s} ({confidence:.3f})")
            formatted_results.append({
                'file': str(file_path),
                'predicted_class': int(pred_class),
                'class_name': class_name,
                'confidence': float(confidence)
            })
        else:
            print(f"{Path(file_path).name:50s} ERROR")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(formatted_results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()
