"""
Generate labels.json from the DAS-BIGORRE dataset README
Automatically extracts earthquake (EQ) and quarry blast (QB) labels
"""

import json
from pathlib import Path


def create_labels_from_readme():
    """
    Create labels.json based on the dataset documentation
    Labels from: DAS-BIGORRE-2022 README
    
    EQ = Earthquake (label: 0)
    QB = Quarry Blast (label: 1)
    """
    
    # Labels from the dataset README
    labels_info = {
        "DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5": ("QB", 0.7),
        "DAS-BIGORRE_2022-09-01_09-19-21_UTC-016.h5": ("QB", 0.8),
        "DAS-BIGORRE_2022-09-03_03-46-50_UTC-010.h5": ("EQ", 0.6),
        "DAS-BIGORRE_2022-09-03_13-06-50_UTC-013.h5": ("EQ", 1.0),
        "DAS-BIGORRE_2022-09-03_18-26-50_UTC-018.h5": ("EQ", 2.4),
        "DAS-BIGORRE_2022-09-05_00-46-50_UTC-007.h5": ("EQ", 1.4),
        "DAS-BIGORRE_2022-09-06_07-56-50_UTC-003.h5": ("EQ", 1.2),
        "DAS-BIGORRE_2022-09-06_10-06-50_UTC-020.h5": ("QB", 1.0),
        "DAS-BIGORRE_2022-09-08_09-56-50_UTC-001.h5": ("QB", 1.1),
        "DAS-BIGORRE_2022-09-08_17-06-50_UTC-004.h5": ("EQ", 0.8),
        "DAS-BIGORRE_2022-09-09_07-06-50_UTC-017.h5": ("EQ", 2.0),
        "DAS-BIGORRE_2022-09-09_17-36-50_UTC-012.h5": ("EQ", 0.4),
        "DAS-BIGORRE_2022-09-12_10-06-50_UTC-002.h5": ("QB", 0.6),
        "DAS-BIGORRE_2022-09-14_09-16-50_UTC-014.h5": ("QB", 1.1),
        "DAS-BIGORRE_2022-09-15_09-06-50_UTC-009.h5": ("EQ", 1.1),
        "DAS-BIGORRE_2022-09-16_11-16-50_UTC-019.h5": ("EQ", 1.6),
        "DAS-BIGORRE_2022-09-16_23-06-50_UTC-008.h5": ("EQ", 1.1),
        "DAS-BIGORRE_2022-09-18_04-16-50_UTC-011.h5": ("EQ", 0.8),
        "DAS-BIGORRE_2022-09-20_04-36-50_UTC-015.h5": ("EQ", 1.3),
    }
    
    # Convert to numeric labels: 0 = Earthquake, 1 = Quarry Blast
    labels = {}
    
    # Check which files actually exist
    data_dir = Path("data")
    existing_files = set()
    
    if data_dir.exists():
        existing_files = {f.name for f in data_dir.glob("*.h5")}
    
    eq_count = 0
    qb_count = 0
    missing_count = 0
    
    for filename, (event_type, magnitude) in labels_info.items():
        # Convert event type to numeric label
        label = 0 if event_type == "EQ" else 1
        labels[filename] = label
        
        # Count and check existence
        if event_type == "EQ":
            eq_count += 1
        else:
            qb_count += 1
        
        if filename not in existing_files and data_dir.exists():
            print(f"⚠ Warning: {filename} not found in data/ directory")
            missing_count += 1
    
    # Save to JSON
    with open('labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print("="*70)
    print("✓ Successfully created labels.json")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Total files:     {len(labels)}")
    print(f"  Earthquakes:     {eq_count} (label: 0)")
    print(f"  Quarry Blasts:   {qb_count} (label: 1)")
    
    if data_dir.exists():
        print(f"\nFile Status:")
        print(f"  Files found:     {len(labels) - missing_count}")
        print(f"  Files missing:   {missing_count}")
        
        if missing_count > 0:
            print(f"\n⚠ Note: {missing_count} labeled files not found in data/ directory")
            print("  Training will use only the files that exist.")
    else:
        print(f"\n⚠ Note: data/ directory not found")
        print("  Labels created but please add data files before training.")
    
    print(f"\nLabel Mapping:")
    print(f"  0 = Earthquake (EQ)")
    print(f"  1 = Quarry Blast (QB)")
    
    print("\n" + "="*70)
    print("Ready to train! Run: python main.py")
    print("="*70)
    
    # Also create a detailed metadata file
    metadata = {
        "dataset": "DAS-BIGORRE-2022",
        "description": "Fiber-Optics Distributed Acoustic Sensing records",
        "location": "Bigorre, Hautes-Pyrénées, France",
        "period": "2022-08-31 to 2022-09-20",
        "total_events": len(labels),
        "earthquakes": eq_count,
        "quarry_blasts": qb_count,
        "label_mapping": {
            "0": "Earthquake",
            "1": "Quarry Blast"
        },
        "files": {}
    }
    
    for filename, (event_type, magnitude) in labels_info.items():
        metadata["files"][filename] = {
            "type": event_type,
            "magnitude": magnitude,
            "label": 0 if event_type == "EQ" else 1
        }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Also created dataset_metadata.json with detailed information")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DAS-BIGORRE-2022 Labels Generator")
    print("="*70)
    print("\nGenerating labels from dataset documentation...")
    print("Source: data/DAS-BIGORRE-2022_ .../README.md\n")
    
    create_labels_from_readme()
