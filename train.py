"""
Training script for DAS seismic classification
Memory-efficient training with gradient accumulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloaders
from models import create_model


class Trainer:
    """
    Memory-efficient trainer with gradient accumulation
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
        
    def prepare_data(self, file_paths, labels):
        """
        Prepare train/val/test splits
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val and test
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            file_paths, labels,
            test_size=self.config['training']['test_split'],
            random_state=42,
            stratify=labels
        )
        
        # Second split: train and val
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=self.config['training']['validation_split'],
            random_state=42,
            stratify=train_val_labels
        )
        
        print(f"Train files: {len(train_files)}")
        print(f"Val files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")
        
        # Get indices for each split
        train_indices = list(range(len(train_files)))
        val_indices = list(range(len(val_files)))
        test_indices = list(range(len(test_files)))
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_files + val_files + test_files,
            train_labels + val_labels + test_labels,
            self.config,
            train_indices,
            [i + len(train_files) for i in val_indices],
            [i + len(train_files) + len(val_files) for i in test_indices]
        )
        
        # Calculate class weights for imbalanced data
        if self.config['training']['use_class_weights']:
            labels_array = np.array(train_labels)
            class_counts = np.bincount(labels_array)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum()
            self.class_weights = torch.FloatTensor(class_weights).to(self.device)
            print(f"Class weights: {self.class_weights}")
        else:
            self.class_weights = None
    
    def build_model(self, num_channels):
        """
        Build model and optimizer
        """
        # Create model
        self.model = create_model(self.config, num_channels)
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
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
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, labels) in enumerate(pbar):
            # Move to device
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
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(data)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # ROC AUC
        all_probs = np.array(all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def test(self):
        """
        Test the model and generate detailed metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Earthquake', 'Quarry Blast']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Earthquake', 'Quarry Blast'],
            yticklabels=['Earthquake', 'Quarry Blast']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix.png')
        print(f"Confusion matrix saved to {self.model_dir / 'confusion_matrix.png'}")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        all_probs = np.array(all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
        
        # Save results
        with open(self.model_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nTest Results:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        return results
    
    def train(self, file_paths, labels):
        """
        Main training loop
        """
        # Prepare data
        self.prepare_data(file_paths, labels)
        
        # Build model
        num_channels = self.train_loader.dataset.num_channels
        self.build_model(num_channels)
        
        # Training loop
        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                torch.save(checkpoint, self.model_dir / 'best_model.pth')
                print(f"✓ Saved best model (accuracy: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # Save checkpoint periodically
            if epoch % self.config['output']['checkpoint_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                torch.save(checkpoint, self.model_dir / f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Clear memory
            torch.cuda.empty_cache()
        
        # Load best model and test
        print("\nLoading best model for testing...")
        checkpoint = torch.load(self.model_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test
        test_results = self.test()
        
        self.writer.close()
        print("\nTraining complete!")
        
        return test_results


if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # You need to provide file_paths and labels
    # Example:
    # file_paths = ['data/file1.h5', 'data/file2.h5', ...]
    # labels = [0, 1, 0, 1, ...]  # 0=earthquake, 1=quarry blast
    
    print("Please create a labels.json file with the format:")
    print('{')
    print('  "file1.h5": 0,  # earthquake')
    print('  "file2.h5": 1,  # quarry blast')
    print('  ...')
    print('}')
