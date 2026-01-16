"""
Train LSTM model on STEAD dataset, test on DAS dataset
This creates a transfer learning scenario where model learns from STEAD
and is evaluated on DAS data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from stead_dataset import create_stead_dataloaders
from dataset import DASDataset, DataLoader
from models import create_model
from train import FocalLoss


class STEADToDASTtrainer:
    """
    Trainer that uses STEAD for training and DAS for testing
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.model_dir = Path(config['output']['model_dir'])
        self.log_dir = Path(config['output']['log_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def prepare_stead_data(self, hdf5_path: str, csv_path: str, target_channels: int):
        """
        Prepare STEAD dataloaders for training/validation
        """
        # Filter criteria for STEAD
        filter_params = {
            'trace_category': 'earthquake_local',  # Focus on local earthquakes
            'source_distance_km': ('<=', 200),      # Within 200 km
            'source_magnitude': ('>', 2.5)          # Magnitude > 2.5
        }
        
        print("\n" + "="*70)
        print("Loading STEAD dataset for training...")
        print("="*70)
        
        self.train_loader, self.val_loader, _ = create_stead_dataloaders(
            hdf5_path=hdf5_path,
            csv_path=csv_path,
            config=self.config,
            target_channels=target_channels,
            train_ratio=0.7,
            val_ratio=0.15,
            filter_params=filter_params
        )
        
        # Calculate class weights
        self._calculate_class_weights(self.train_loader)
        
    def prepare_das_test_data(self, file_paths: List[str], labels: List[int]):
        """
        Prepare DAS dataloader for testing only
        """
        print("\n" + "="*70)
        print("Loading DAS dataset for testing...")
        print("="*70)
        
        test_dataset = DASDataset(
            file_paths,
            labels,
            window_size=self.config['data']['window_size'],
            stride=self.config['data']['stride'],
            num_channels=self.config['data']['num_channels'],
            transform=None
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=0,  # Single process for test
            pin_memory=True
        )
        
        print(f"DAS test windows: {len(test_dataset):,}")
        
        # Verify channel count matches
        sample_data, _ = test_dataset[0]
        das_channels = sample_data.shape[0]
        print(f"DAS channels: {das_channels}")
        
        return das_channels
        
    def _calculate_class_weights(self, train_loader):
        """Calculate class weights from training data"""
        if not self.config['training']['use_class_weights']:
            self.class_weights = None
            return
        
        # Count labels in training set
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        labels_array = np.array(all_labels)
        class_counts = np.bincount(labels_array)
        
        # Calculate weights
        total_samples = len(labels_array)
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)
        class_weights = class_weights * num_classes / class_weights.sum()
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"\nClass distribution in STEAD training data:")
        print(f"  Earthquakes (0): {class_counts[0]} samples, weight: {class_weights[0]:.4f}")
        print(f"  Noise (1): {class_counts[1]} samples, weight: {class_weights[1]:.4f}")
        
    def build_model(self, num_channels: int):
        """Build model and optimizer"""
        self.model = create_model(self.config, num_channels)
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {num_params:,}")
        
        # Loss function
        use_focal_loss = self.config['training'].get('use_focal_loss', False)
        if use_focal_loss:
            focal_gamma = self.config['training'].get('focal_gamma', 2.0)
            self.criterion = FocalLoss(alpha=self.class_weights, gamma=focal_gamma)
            print(f"Using Focal Loss (gamma={focal_gamma})")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print("Using Cross Entropy Loss")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
    def train_epoch(self, epoch: int):
        """Train for one epoch on STEAD data"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Train/F1', f1, epoch)
        
        return avg_loss, accuracy, f1
        
    def validate_epoch(self, epoch: int):
        """Validate on STEAD validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc="Validation"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(data)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)
        
        return avg_loss, accuracy, f1
        
    def test_on_das(self):
        """Test model on DAS dataset"""
        print("\n" + "="*70)
        print("Testing on DAS dataset...")
        print("="*70)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="Testing on DAS"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # ROC-AUC
        all_probs = np.array(all_probs)
        if len(np.unique(all_labels)) == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Print results
        print("\n" + "="*70)
        print("DAS Test Results:")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=['Earthquake', 'Quarry Blast']))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, ['Earthquake', 'Quarry Blast'])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - DAS Test Set')
        
        save_path = self.model_dir / 'confusion_matrix_das_test.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
        
    def train(self, num_epochs: int):
        """Full training loop"""
        print("\n" + "="*70)
        print("Starting training on STEAD dataset...")
        print("="*70)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch(epoch)
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config
                }, self.model_dir / 'best_model_stead.pth')
                print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("="*70)
        
        # Load best model for testing
        checkpoint = torch.load(self.model_dir / 'best_model_stead.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("\nLoaded best model for DAS testing")


def main():
    import sys
    
    # Load configuration
    config_file = 'config.yaml'
    if not Path(config_file).exists():
        print(f"Error: {config_file} not found!")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # STEAD dataset paths (user needs to download these)
    stead_hdf5 = Path('data/stead/merge.hdf5')
    stead_csv = Path('data/stead/merge.csv')
    
    if not stead_hdf5.exists() or not stead_csv.exists():
        print("\n" + "="*70)
        print("ERROR: STEAD dataset not found!")
        print("="*70)
        print("\nPlease download STEAD dataset:")
        print("1. Download from: https://rebrand.ly/whole")
        print("2. Extract to: data/stead/")
        print("3. Files needed: merge.hdf5 and merge.csv")
        print("\nSee README_STEAD.md for more details")
        return
    
    # Load DAS labels for testing
    labels_file = 'labels.json'
    if not Path(labels_file).exists():
        print(f"Error: {labels_file} not found!")
        print("Run: python generate_labels.py")
        return
    
    with open(labels_file, 'r') as f:
        das_labels_dict = json.load(f)
    
    # Prepare DAS file paths
    das_file_paths = []
    das_labels = []
    data_dir = Path('data')
    
    for filename, label in das_labels_dict.items():
        file_path = data_dir / filename
        if file_path.exists():
            das_file_paths.append(str(file_path))
            das_labels.append(label)
    
    print(f"\nDAS files for testing: {len(das_file_paths)}")
    
    # Create trainer
    trainer = STEADToDASTtrainer(config)
    
    # Get DAS channel count first
    temp_das_dataset = DASDataset(
        das_file_paths[:1],  # Just load first file
        [das_labels[0]],
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        num_channels=config['data']['num_channels']
    )
    sample_data, _ = temp_das_dataset[0]
    das_channels = sample_data.shape[0]
    print(f"\nDetected DAS channels: {das_channels}")
    
    # Prepare STEAD data (adapt to DAS channel count)
    trainer.prepare_stead_data(
        hdf5_path=str(stead_hdf5),
        csv_path=str(stead_csv),
        target_channels=das_channels
    )
    
    # Prepare DAS test data
    trainer.prepare_das_test_data(das_file_paths, das_labels)
    
    # Build model
    trainer.build_model(num_channels=das_channels)
    
    # Train on STEAD
    trainer.train(num_epochs=config['training']['epochs'])
    
    # Test on DAS
    test_results = trainer.test_on_das()
    
    # Save results
    results_file = trainer.model_dir / 'das_test_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        results_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else float(v)
            for k, v in test_results.items()
            if k != 'confusion_matrix'
        }
        json.dump(results_json, f, indent=2)
    print(f"\nSaved test results to {results_file}")


if __name__ == '__main__':
    main()
