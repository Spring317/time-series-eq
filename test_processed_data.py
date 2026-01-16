"""
Test script to verify processed data compatibility
"""
import h5py
import yaml
import json
from pathlib import Path
from dataset import DASDataset


def test_processed_file_structure():
    """Test that processed files have the expected structure"""
    print("="*70)
    print("Test 1: Checking processed file structure")
    print("="*70)
    
    processed_dir = Path('data/processed_chunks')
    if not processed_dir.exists():
        print("❌ ERROR: processed_chunks directory not found")
        return False
    
    # Get first processed file
    files = list(processed_dir.glob('*_processed.h5'))
    if not files:
        print("❌ ERROR: No processed files found")
        return False
    
    test_file = files[0]
    print(f"Testing file: {test_file.name}")
    
    try:
        with h5py.File(test_file, 'r') as f:
            # Check for required keys
            required_keys = ['data', 'distance', 'time']
            for key in required_keys:
                if key not in f:
                    print(f"❌ ERROR: Missing key '{key}' in processed file")
                    return False
                print(f"  ✓ Found '{key}': shape={f[key].shape}, dtype={f[key].dtype}")
            
            # Check data shape
            data_shape = f['data'].shape
            if len(data_shape) != 2:
                print(f"❌ ERROR: Data should be 2D, got shape {data_shape}")
                return False
            
            print(f"  ✓ Data has valid 2D shape: {data_shape}")
            
        print("✓ Processed file structure is valid\n")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_dataset_loading():
    """Test that DASDataset can load processed files"""
    print("="*70)
    print("Test 2: Testing dataset loading with processed files")
    print("="*70)
    
    # Load labels
    labels_file = Path('labels.json')
    if not labels_file.exists():
        print("❌ ERROR: labels.json not found")
        return False
    
    with open(labels_file, 'r') as f:
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
    
    if not file_paths:
        print("❌ ERROR: No matching processed files found")
        return False
    
    print(f"Found {len(file_paths)} processed files")
    print(f"  Earthquakes: {labels.count(0)}")
    print(f"  Quarry blasts: {labels.count(1)}")
    
    # Create dataset
    try:
        dataset = DASDataset(
            file_paths=file_paths[:2],  # Test with first 2 files
            labels=labels[:2],
            window_size=1024,
            stride=512,
            cache_file_info=False
        )
        
        print(f"  ✓ Dataset created with {len(dataset)} windows")
        print(f"  ✓ Number of channels: {dataset.num_channels}")
        
        # Test loading a sample
        data, label = dataset[0]
        print(f"  ✓ Loaded sample: data shape={data.shape}, label={label}")
        
        if data.shape[1] != 1024:
            print(f"❌ ERROR: Expected window size 1024, got {data.shape[1]}")
            return False
        
        print("✓ Dataset loading successful\n")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_compatibility():
    """Test that config.yaml is compatible"""
    print("="*70)
    print("Test 3: Checking config.yaml compatibility")
    print("="*70)
    
    config_file = Path('config.yaml')
    if not config_file.exists():
        print("❌ ERROR: config.yaml not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'model', 'training', 'output']
        for section in required_sections:
            if section not in config:
                print(f"❌ ERROR: Missing section '{section}' in config")
                return False
            print(f"  ✓ Found section '{section}'")
        
        # Check data parameters
        data_config = config['data']
        print(f"  ✓ Window size: {data_config.get('window_size')}")
        print(f"  ✓ Stride: {data_config.get('stride')}")
        print(f"  ✓ Batch size: {data_config.get('batch_size')}")
        
        # Check model type
        model_type = config['model'].get('type', 'cnn')
        print(f"  ✓ Model type: {model_type}")
        
        print("✓ Config is valid\n")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PROCESSED DATA COMPATIBILITY TEST")
    print("="*70 + "\n")
    
    tests = [
        test_processed_file_structure,
        test_dataset_loading,
        test_config_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now train with processed data:")
        print("  python main.py --processed")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check the errors above")
    
    return all(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
