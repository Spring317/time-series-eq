#!/usr/bin/env python3
"""
Train LightEQ-style model on STEAD, test on DAS

This script implements the state-of-the-art preprocessing from LightEQ:
1. STFT spectrogram transformation
2. Data augmentation (shift, noise, scale)
3. CNN+LSTM architecture

Usage:
    # Use pre-processed numpy files (recommended - much faster!)
    python train_lighteq.py --mode full --stead-dir /storage/student8/STEAD
    
    # Process from raw HDF5 (slower, needs raw STEAD files)
    python train_lighteq.py --mode full --from-raw
    
    python train_lighteq.py --mode test
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Local imports
from load_preprocessed_stead import (
    load_preprocessed_stead,
    create_dataloaders_from_preprocessed,
    PreprocessedSTEADDataset
)
from lighteq_models import get_lighteq_model

# Optional: for raw processing (if --from-raw is used)
try:
    from stead_lighteq_dataset import (
        STEADLightEQDataset,
        DASLightEQDataset,
        create_stead_lighteq_dataloaders,
        prepare_stead_splits
    )
    HAS_RAW_PROCESSING = True
except ImportError:
    HAS_RAW_PROCESSING = False


class LightEQTrainer:
    """
    Trainer for LightEQ-style model
    
    Train on STEAD, test on DAS with STFT preprocessing
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = None
    ):
        """
        Args:
            config: Configuration dictionary
            device: Device to use (cuda/cpu)
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create directories
        self.log_dir = Path(config.get('log_dir', 'logs/lighteq'))
        self.model_dir = Path(config.get('model_dir', 'models/lighteq'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epoch = 0
    
    def _create_model(self) -> nn.Module:
        """Create LightEQ model"""
        model_type = self.config.get('model_type', 'medium')
        num_classes = self.config.get('num_classes', 2)
        
        print(f"Creating LightEQ model: {model_type}")
        
        return get_lighteq_model(
            model_type=model_type,
            num_classes=num_classes,
            dropout=self.config.get('dropout', 0.25)
        )
    
    def prepare_stead_data(self, from_raw: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare STEAD training and validation data
        
        Args:
            from_raw: If True, process from raw HDF5. If False, load pre-processed numpy files.
        
        Returns:
            train_loader, val_loader
        """
        stead_config = self.config.get('stead', {})
        batch_size = self.config.get('batch_size', 64)
        num_workers = self.config.get('num_workers', 4)
        
        if from_raw:
            # Process from raw HDF5 files
            if not HAS_RAW_PROCESSING:
                raise ImportError("Raw processing requires stead_lighteq_dataset.py")
            
            hdf5_path = stead_config.get('hdf5_path', 'data/stead/merge.hdf5')
            csv_path = stead_config.get('csv_path', 'data/stead/merge.csv')
            
            if not Path(hdf5_path).exists():
                raise FileNotFoundError(f"STEAD HDF5 not found: {hdf5_path}")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"STEAD CSV not found: {csv_path}")
            
            print(f"Loading STEAD from raw files: {hdf5_path}")
            
            # Prepare splits
            train_traces, val_traces, _ = prepare_stead_splits(
                csv_path=csv_path,
                train_ratio=stead_config.get('train_ratio', 0.85),
                val_ratio=stead_config.get('val_ratio', 0.10),
                test_ratio=stead_config.get('test_ratio', 0.05),
                max_samples=stead_config.get('max_samples_per_class', None),
                balance_classes=stead_config.get('balance_classes', True)
            )
            
            # Create datasets
            train_dataset = STEADLightEQDataset(
                hdf5_path=hdf5_path,
                csv_path=csv_path,
                trace_list=train_traces,
                norm_mode=self.config.get('norm_mode', 'max'),
                augmentation=True,
                shift_event_r=stead_config.get('shift_event_r', 0.9),
                add_event_r=stead_config.get('add_event_r', 0.5),
                add_noise_r=stead_config.get('add_noise_r', 0.4),
                scale_amplitude_r=stead_config.get('scale_amplitude_r', 0.3)
            )
            
            val_dataset = STEADLightEQDataset(
                hdf5_path=hdf5_path,
                csv_path=csv_path,
                trace_list=val_traces,
                norm_mode=self.config.get('norm_mode', 'max'),
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
        else:
            # Load pre-processed numpy files (FAST!)
            stead_dir = stead_config.get('preprocessed_dir', '/storage/student8/STEAD')
            prefix = stead_config.get('file_prefix', '76')
            
            print(f"Loading pre-processed STEAD from: {stead_dir}")
            print(f"File prefix: {prefix}")
            
            train_loader, val_loader, self.stead_test_loader = create_dataloaders_from_preprocessed(
                data_dir=stead_dir,
                prefix=prefix,
                batch_size=batch_size,
                val_split=stead_config.get('val_split', 0.1),
                num_workers=num_workers
            )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def prepare_das_test_data(self) -> DataLoader:
        """
        Prepare DAS test data with LightEQ preprocessing
        
        Returns:
            test_loader
        """
        das_config = self.config.get('das', {})
        data_dir = Path(das_config.get('data_dir', 'data'))
        
        # Load labels
        labels_path = das_config.get('labels_path', 'labels.json')
        if Path(labels_path).exists():
            with open(labels_path) as f:
                labels_data = json.load(f)
        else:
            print("Warning: labels.json not found, using default labels")
            labels_data = {}
        
        # Find DAS files
        das_files = sorted(data_dir.glob('DAS-*.h5'))
        
        file_paths = []
        labels = []
        
        for file_path in das_files:
            file_name = file_path.name
            
            # Get label (0 = earthquake, 1 = quarry blast)
            label = 0  # Default to earthquake
            if labels_data:
                for event_type, files in labels_data.items():
                    if file_name in files:
                        label = 0 if event_type in ['earthquake', 'local_earthquake'] else 1
                        break
            
            file_paths.append(str(file_path))
            labels.append(label)
        
        print(f"Found {len(file_paths)} DAS files")
        print(f"  Label distribution: {sum(1 for l in labels if l == 0)} earthquakes, {sum(1 for l in labels if l == 1)} other")
        
        # Create dataset
        test_dataset = DASLightEQDataset(
            file_paths=file_paths,
            labels=labels,
            window_size=das_config.get('window_size', 12000),  # DAS at 200Hz
            stride=das_config.get('stride', 6000),
            n_channels=3,  # Match STEAD
            norm_mode=self.config.get('norm_mode', 'max')
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        
        early_stop_patience = self.config.get('early_stop_patience', 15)
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self._save_checkpoint('best_model.pth')
                print(f"  New best model saved! (Val Loss: {self.best_val_loss:.4f})")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        print(f"\nTraining complete! Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")
        
        # Save final model
        self._save_checkpoint('final_model.pth')
    
    def test_on_das(self, test_loader: DataLoader = None) -> Dict:
        """
        Test model on DAS data
        
        Args:
            test_loader: DAS test dataloader (creates one if None)
        
        Returns:
            Test metrics
        """
        if test_loader is None:
            test_loader = self.prepare_das_test_data()
        
        # Load best model
        best_model_path = self.model_dir / 'best_model.pth'
        if best_model_path.exists():
            self._load_checkpoint('best_model.pth')
            print("Loaded best model for testing")
        
        print("\nTesting on DAS data...")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc='Testing on DAS'):
                data = data.to(self.device)
                
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        # Print results
        print("\n" + "="*60)
        print("DAS Test Results (LightEQ Preprocessing)")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Earthquake', 'Quarry Blast']
        ))
        
        # Save results
        results_path = self.model_dir / 'das_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard"""
        for name, value in train_metrics.items():
            self.writer.add_scalar(f'train/{name}', value, self.epoch)
        
        for name, value in val_metrics.items():
            self.writer.add_scalar(f'val/{name}', value, self.epoch)
        
        # Log learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/lr', lr, self.epoch)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        path = self.model_dir / filename
        torch.save(checkpoint, path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.model_dir / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration file"""
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Set defaults for LightEQ training
    defaults = {
        'model_type': 'medium',  # 'light', 'medium', 'full'
        'num_classes': 2,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout': 0.25,
        'num_epochs': 50,
        'early_stop_patience': 15,
        'num_workers': 4,
        'norm_mode': 'max',
        'log_dir': 'logs/lighteq',
        'model_dir': 'models/lighteq',
        'stead': {
            # Pre-processed numpy files (FAST - recommended!)
            'preprocessed_dir': '/storage/student8/STEAD',
            'file_prefix': '76',
            'val_split': 0.1,
            # Raw HDF5 files (only if --from-raw is used)
            'hdf5_path': 'data/stead/merge.hdf5',
            'csv_path': 'data/stead/merge.csv',
            'train_ratio': 0.85,
            'val_ratio': 0.10,
            'test_ratio': 0.05,
            'max_samples_per_class': None,
            'balance_classes': True,
            'shift_event_r': 0.9,
            'add_event_r': 0.5,
            'add_noise_r': 0.4,
            'scale_amplitude_r': 0.3
        },
        'das': {
            'data_dir': 'data',
            'labels_path': 'labels.json',
            'window_size': 12000,  # 60 seconds at 200Hz
            'stride': 6000
        }
    }
    
    # Merge with loaded config
    def merge_dicts(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    return merge_dicts(defaults, config)


def main():
    parser = argparse.ArgumentParser(
        description='Train LightEQ-style model on STEAD, test on DAS'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test', 'full'],
        default='full',
        help='Mode: train (STEAD only), test (DAS only), full (train + test)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--stead-dir',
        default=None,
        help='Path to pre-processed STEAD directory (default: /storage/student8/STEAD)'
    )
    parser.add_argument(
        '--from-raw',
        action='store_true',
        help='Process from raw HDF5 instead of pre-processed numpy files'
    )
    parser.add_argument(
        '--model-type',
        choices=['light', 'medium', 'full'],
        default=None,
        help='Model type (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max samples per class for faster training'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.stead_dir:
        config['stead']['preprocessed_dir'] = args.stead_dir
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_samples:
        config['stead']['max_samples_per_class'] = args.max_samples
    
    # Print config
    print("\nConfiguration:")
    print(json.dumps(config, indent=2, default=str))
    
    # Create trainer
    trainer = LightEQTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer._load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")
    
    # Run based on mode
    if args.mode in ['train', 'full']:
        # Prepare STEAD data (use pre-processed by default)
        train_loader, val_loader = trainer.prepare_stead_data(from_raw=args.from_raw)
        
        # Train
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['num_epochs']
        )
    
    if args.mode in ['test', 'full']:
        # Test on DAS
        trainer.test_on_das()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
