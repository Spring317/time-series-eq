"""
Main script to run the training pipeline
"""

import yaml
import json
from pathlib import Path
from train import Trainer


def load_labels(labels_file):
    """
    Load labels from JSON file
    Returns file_paths and labels lists
    """
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)
    
    file_paths = []
    labels = []
    
    for filename, label in labels_dict.items():
        file_path = Path('data') / filename
        if file_path.exists():
            file_paths.append(str(file_path))
            labels.append(label)
        else:
            print(f"Warning: {file_path} not found, skipping...")
    
    return file_paths, labels


def main():
    # Load configuration
    config_file = 'config.yaml'
    labels_file = 'labels.json'
    
    if not Path(config_file).exists():
        print(f"Error: {config_file} not found!")
        print("Please create config.yaml first.")
        return
    
    if not Path(labels_file).exists():
        print(f"Error: {labels_file} not found!")
        print("")
        print("This dataset (DAS-BIGORRE-2022) has predefined labels:")
        print("  - 13 Earthquakes")
        print("  - 6 Quarry Blasts")
        print("")
        print("Generate labels automatically by running:")
        print("  python generate_labels.py")
        return
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load labels
    print("Loading labels...")
    file_paths, labels = load_labels(labels_file)
    
    print(f"\nDataset summary:")
    print(f"Total files: {len(file_paths)}")
    print(f"Earthquakes: {labels.count(0)}")
    print(f"Quarry blasts: {labels.count(1)}")
    
    if len(file_paths) == 0:
        print("\nError: No valid files found!")
        return
    
    # Create trainer and train
    print("\nInitializing trainer...")
    trainer = Trainer(config, device='cuda')
    
    print("\nStarting training...")
    results = trainer.train(file_paths, labels)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print("\nFinal test results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nModel saved to: {config['output']['model_dir']}")
    print(f"Logs saved to: {config['output']['log_dir']}")
    print("\nTo view training logs, run:")
    print(f"  tensorboard --logdir {config['output']['log_dir']}")


if __name__ == '__main__':
    main()
