"""
Generate labels.json from the DAS-BIGORRE dataset README
Automatically extracts earthquake (EQ) and quarry blast (QB) labels
"""

import json
import re
from pathlib import Path


def parse_readme_file():
    """
    Parse the README.md file to extract event labels and magnitudes
    
    Returns:
        dict: Mapping of filename -> (event_type, magnitude)
    """
    readme_path = Path("data/DAS-BIGORRE-2022_ Fiber-Optics Distributed Acoustic Sensing records (Bigorre, Hautes-Pyrénées, France)/README.md")
    
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found at: {readme_path}")
    
    labels_info = {}
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse lines with format: - DAS-BIGORRE_2022-XX-XX_XX-XX-XX_UTC.h5 : QB/EQ, Mw=X.X, ...
    pattern = r'- (DAS-BIGORRE_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC)\.h5\s*:\s*(EQ|QB),\s*Mw=([\d.]+)'
    
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} labeled events in README.md")
    
    # Map the base filename to include the suffix that matches actual files
    data_dir = Path("data")
    existing_files = {}
    
    if data_dir.exists():
        for f in data_dir.glob("*.h5"):
            # Extract base name without suffix (like -005, -016, etc.)
            base = re.match(r'(DAS-BIGORRE_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC)', f.name)
            if base:
                existing_files[base.group(1)] = f.name
    
    for base_name, event_type, magnitude in matches:
        # Find the actual filename with suffix
        if base_name in existing_files:
            actual_filename = existing_files[base_name]
            labels_info[actual_filename] = (event_type, float(magnitude))
            print(f"  {actual_filename}: {event_type}, Mw={magnitude}")
        else:
            # If file doesn't exist yet, use base name with .h5
            filename = f"{base_name}.h5"
            labels_info[filename] = (event_type, float(magnitude))
            print(f"  {filename}: {event_type}, Mw={magnitude} (not found in data/)")
    
    return labels_info


def create_labels_from_readme():
    """
    Create labels.json based on the dataset documentation
    Labels from: DAS-BIGORRE-2022 README
    
    EQ = Earthquake (label: 0)
    QB = Quarry Blast (label: 1)
    """
    
    # Parse README.md to extract labels
    print("Parsing README.md file...")
    labels_info = parse_readme_file()
    
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
