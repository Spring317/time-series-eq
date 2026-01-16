"""
Test DataLoader with multiple workers to catch memory issues
"""
import yaml
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import DASDataset


def test_dataloader_batching():
    """Test that DataLoader can properly batch data from processed files"""
    print("="*70)
    print("Testing DataLoader with Multiple Workers")
    print("="*70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load labels
    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)
    
    # Get processed files
    processed_dir = Path('data/processed_chunks')
    file_paths = []
    labels = []
    
    for filename, label in labels_dict.items():
        processed_name = filename.replace('.h5', '_processed.h5')
        file_path = processed_dir / processed_name
        if file_path.exists():
            file_paths.append(str(file_path))
            labels.append(label)
    
    if len(file_paths) < 2:
        print("❌ ERROR: Need at least 2 files for testing")
        return False
    
    print(f"Testing with {len(file_paths[:3])} files...")
    
    # Create dataset with first 3 files
    dataset = DASDataset(
        file_paths=file_paths[:3],
        labels=labels[:3],
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        cache_file_info=False
    )
    
    print(f"  Dataset has {len(dataset)} windows")
    
    # Test with multiple workers (this is where the error occurs)
    num_workers = config['data']['num_workers']
    print(f"  Testing with {num_workers} workers...")
    
    try:
        # Test regular DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False  # Important: disabled to avoid issues
        )
        
        # Try to iterate through batches
        for i, (data, label) in enumerate(loader):
            print(f"  Batch {i+1}: data shape={data.shape}, labels shape={label.shape}")
            
            # Verify shapes
            assert data.dim() == 3, f"Expected 3D data, got {data.dim()}D"
            assert data.shape[1] == dataset.num_channels, f"Channel mismatch"
            assert data.shape[2] == config['data']['window_size'], f"Window size mismatch"
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("  ✓ DataLoader batching works correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_sample_loading():
    """Test loading individual samples"""
    print("\n" + "="*70)
    print("Testing Single Sample Loading")
    print("="*70)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load labels
    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)
    
    # Get first processed file
    processed_dir = Path('data/processed_chunks')
    file_path = None
    label = None
    
    for filename, lbl in labels_dict.items():
        processed_name = filename.replace('.h5', '_processed.h5')
        fp = processed_dir / processed_name
        if fp.exists():
            file_path = str(fp)
            label = lbl
            break
    
    if file_path is None:
        print("❌ ERROR: No processed files found")
        return False
    
    print(f"Testing with file: {Path(file_path).name}")
    
    try:
        # Create dataset
        dataset = DASDataset(
            file_paths=[file_path],
            labels=[label],
            window_size=config['data']['window_size'],
            stride=config['data']['stride'],
            cache_file_info=False
        )
        
        # Load multiple samples
        for i in range(min(5, len(dataset))):
            data, lbl = dataset[i]
            print(f"  Sample {i}: shape={data.shape}, dtype={data.dtype}, label={lbl}")
            
            # Verify tensor properties
            assert isinstance(data, torch.Tensor), "Data should be torch.Tensor"
            assert data.dtype == torch.float32, f"Expected float32, got {data.dtype}"
            assert data.is_contiguous(), "Data should be contiguous"
            
        print("  ✓ Single sample loading works correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all DataLoader tests"""
    print("\n" + "="*70)
    print("DATALOADER COMPATIBILITY TEST")
    print("="*70 + "\n")
    
    tests = [
        test_single_sample_loading,
        test_dataloader_batching
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL DATALOADER TESTS PASSED!")
        print("\nThe storage resize error should now be fixed.")
        print("Try training again with: python main.py --processed")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check the errors above for more details")
    
    return all(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
