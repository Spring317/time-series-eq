"""
Create labels.json from README.md information
Automatically maps files to earthquake (0) or quarry blast (1) labels
"""
import json
from pathlib import Path


def create_labels_from_readme():
    """
    Create labels based on the dataset README
    """
    # Mapping from README.md
    # EQ = Earthquake (0), QB = Quarry Blast (1)
    
    file_labels = {
        'DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5': ('QB', 1, 0.7),
        'DAS-BIGORRE_2022-09-01_09-19-21_UTC-016.h5': ('QB', 1, 0.8),
        'DAS-BIGORRE_2022-09-03_03-46-50_UTC-010.h5': ('EQ', 0, 0.6),
        'DAS-BIGORRE_2022-09-03_13-06-50_UTC-013.h5': ('EQ', 0, 1.0),
        'DAS-BIGORRE_2022-09-03_18-26-50_UTC-018.h5': ('EQ', 0, 2.4),
        'DAS-BIGORRE_2022-09-05_00-46-50_UTC-007.h5': ('EQ', 0, 1.4),
        'DAS-BIGORRE_2022-09-06_07-56-50_UTC-003.h5': ('EQ', 0, 1.2),
        'DAS-BIGORRE_2022-09-06_10-06-50_UTC-020.h5': ('QB', 1, 1.0),
        'DAS-BIGORRE_2022-09-08_09-56-50_UTC-001.h5': ('QB', 1, 1.1),
        'DAS-BIGORRE_2022-09-08_17-06-50_UTC-004.h5': ('EQ', 0, 0.8),
        'DAS-BIGORRE_2022-09-09_07-06-50_UTC-017.h5': ('EQ', 0, 2.0),
        'DAS-BIGORRE_2022-09-09_17-36-50_UTC-012.h5': ('EQ', 0, 0.4),
        'DAS-BIGORRE_2022-09-12_10-06-50_UTC-002.h5': ('QB', 1, 0.6),
        'DAS-BIGORRE_2022-09-14_09-16-50_UTC-014.h5': ('QB', 1, 1.1),
        'DAS-BIGORRE_2022-09-15_09-06-50_UTC-009.h5': ('EQ', 0, 1.1),
        'DAS-BIGORRE_2022-09-16_11-16-50_UTC-019.h5': ('EQ', 0, 1.6),
        'DAS-BIGORRE_2022-09-16_23-06-50_UTC-008.h5': ('EQ', 0, 1.1),
        'DAS-BIGORRE_2022-09-18_04-16-50_UTC-011.h5': ('EQ', 0, 0.8),
        'DAS-BIGORRE_2022-09-20_04-36-50_UTC-015.h5': ('EQ', 0, 1.3),
    }
    
    # Check which files exist (original or processed)
    labels_dict = {}
    
    # For processed files only
    processed_dir = Path('data/processed_chunks')
    if processed_dir.exists():
        for filename, (event_type, label, magnitude) in file_labels.items():
            processed_filename = filename.replace('.h5', '_processed.h5')
            if (processed_dir / processed_filename).exists():
                labels_dict[processed_filename] = label
    
    # If no processed files found, use original files
    if not labels_dict:
        data_dir = Path('data')
        for filename, (event_type, label, magnitude) in file_labels.items():
            if (data_dir / filename).exists():
                labels_dict[filename] = label
    
    # Save to JSON
    output_file = 'labels.json'
    with open(output_file, 'w') as f:
        json.dump(labels_dict, f, indent=2, sort_keys=True)
    
    # Print summary
    print("="*70)
    print("LABELS CREATED FROM README.md")
    print("="*70)
    print(f"\nTotal files labeled: {len(labels_dict)}")
    
    # Count by type
    earthquakes = sum(1 for label in labels_dict.values() if label == 0)
    quarry_blasts = sum(1 for label in labels_dict.values() if label == 1)
    
    print(f"\nLabel distribution:")
    print(f"  Earthquakes (0): {earthquakes}")
    print(f"  Quarry blasts (1): {quarry_blasts}")
    
    print(f"\nLabels saved to: {output_file}")
    
    # Show detailed mapping
    print("\nDetailed mapping:")
    print("-" * 70)
    for filename in sorted(labels_dict.keys()):
        label = labels_dict[filename]
        event_type = "Earthquake" if label == 0 else "Quarry Blast"
        
        # Get original filename to show magnitude
        orig_filename = filename.replace('_processed.h5', '.h5')
        if orig_filename in file_labels:
            _, _, magnitude = file_labels[orig_filename]
            print(f"{filename:60s} | {event_type:13s} | Mw={magnitude}")
        else:
            print(f"{filename:60s} | {event_type:13s}")
    
    return labels_dict


def verify_labels():
    """
    Verify that all processed files have labels
    """
    processed_dir = Path('data/processed_chunks')
    processed_files = sorted(processed_dir.glob('*_processed.h5'))
    
    if not Path('labels.json').exists():
        print("Error: labels.json not found. Run this script first.")
        return
    
    with open('labels.json', 'r') as f:
        labels = json.load(f)
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check for unlabeled files
    unlabeled = []
    for f in processed_files:
        if f.name not in labels:
            unlabeled.append(f.name)
    
    if unlabeled:
        print(f"\n⚠️  Warning: {len(unlabeled)} processed files without labels:")
        for f in unlabeled:
            print(f"  - {f}")
    else:
        print(f"\n✅ All {len(processed_files)} processed files are labeled!")
    
    # Check for missing files
    missing = []
    for filename in labels.keys():
        file_path = processed_dir / filename
        if not file_path.exists():
            # Try original data directory
            orig_path = Path('data') / filename
            if not orig_path.exists():
                missing.append(filename)
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} labeled files not found:")
        for f in missing[:5]:
            print(f"  - {f}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print("✅ All labeled files exist!")


if __name__ == '__main__':
    # Create labels
    labels_dict = create_labels_from_readme()
    
    # Verify
    verify_labels()
    
    print("\n" + "="*70)
    print("✅ READY TO TRAIN!")
    print("="*70)
    print("\nRun training with:")
    print("  python main.py --processed")
