"""
Quick test to verify the pipeline is working
"""

import yaml
from pathlib import Path
import json

def test_pipeline():
    print("="*70)
    print("Pipeline Test")
    print("="*70)
    
    # 1. Check labels
    print("\n1. Checking labels.json...")
    if Path('labels.json').exists():
        with open('labels.json', 'r') as f:
            labels = json.load(f)
        print(f"   ✓ Found {len(labels)} labeled files")
        eq_count = sum(1 for v in labels.values() if v == 0)
        qb_count = sum(1 for v in labels.values() if v == 1)
        print(f"   ✓ {eq_count} Earthquakes, {qb_count} Quarry Blasts")
    else:
        print("   ✗ labels.json not found!")
        return False
    
    # 2. Check config
    print("\n2. Checking config.yaml...")
    if Path('config.yaml').exists():
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config loaded")
        print(f"   ✓ Model type: {config['model']['type']}")
        print(f"   ✓ Batch size: {config['data']['batch_size']}")
        print(f"   ✓ Window size: {config['data']['window_size']}")
    else:
        print("   ✗ config.yaml not found!")
        return False
    
    # 3. Check data files
    print("\n3. Checking data files...")
    data_dir = Path('data')
    h5_files = list(data_dir.glob('*.h5'))
    print(f"   ✓ Found {len(h5_files)} HDF5 files")
    
    # 4. Test loading one file
    print("\n4. Testing HDF5 file loading...")
    try:
        import h5py
        test_file = h5_files[0]
        with h5py.File(test_file, 'r') as f:
            data = f['python_processing/sr/data']
            shape = data.shape
            print(f"   ✓ Successfully opened {test_file.name}")
            print(f"   ✓ Data shape: {shape}")
            print(f"   ✓ Channels: {shape[0]}, Time samples: {shape[1]}")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        return False
    
    # 5. Test dataset creation
    print("\n5. Testing dataset creation...")
    try:
        from dataset import DASDataset
        file_paths = [str(f) for f in h5_files[:2]]  # Test with 2 files
        labels_list = [labels[f.name] for f in h5_files[:2]]
        
        dataset = DASDataset(
            file_paths,
            labels_list,
            window_size=config['data']['window_size'],
            stride=config['data']['stride'],
            cache_file_info=False  # Don't cache for test
        )
        
        print(f"   ✓ Dataset created")
        print(f"   ✓ Total windows: {len(dataset)}")
        print(f"   ✓ Number of channels: {dataset.num_channels}")
        
        # Test loading one sample
        data, label = dataset[0]
        print(f"   ✓ Sample shape: {data.shape}")
        print(f"   ✓ Sample label: {label.item()}")
        
    except Exception as e:
        print(f"   ✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test model creation
    print("\n6. Testing model creation...")
    try:
        from models import create_model
        model = create_model(config, dataset.num_channels)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created: {config['model']['type']}")
        print(f"   ✓ Parameters: {num_params:,}")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ All tests passed! Pipeline is ready to train.")
    print("="*70)
    print("\nRun: python main.py")
    
    return True


if __name__ == '__main__':
    test_pipeline()
