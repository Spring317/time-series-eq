"""
Helper script to create labels.json file
Assists in labeling your DAS data files as earthquake (0) or quarry blast (1)
"""

import json
from pathlib import Path
import h5py


def scan_data_directory(data_dir='data'):
    """
    Scan data directory for HDF5 files
    """
    data_path = Path(data_dir)
    h5_files = []
    
    # Find all .h5 files
    for file_path in sorted(data_path.glob('*.h5')):
        h5_files.append(file_path.name)
    
    return h5_files


def create_labels_interactively():
    """
    Interactive labeling of files
    """
    print("Scanning data directory...")
    files = scan_data_directory()
    
    if len(files) == 0:
        print("No HDF5 files found in data/ directory!")
        return
    
    print(f"Found {len(files)} HDF5 files\n")
    print("Label each file:")
    print("  0 = Earthquake")
    print("  1 = Quarry Blast")
    print("  s = Skip this file")
    print("  q = Quit and save\n")
    
    labels = {}
    
    for i, filename in enumerate(files):
        while True:
            response = input(f"[{i+1}/{len(files)}] {filename}: ")
            
            if response.lower() == 'q':
                print("\nQuitting...")
                save_labels(labels)
                return
            elif response.lower() == 's':
                print("Skipping...")
                break
            elif response == '0':
                labels[filename] = 0
                print("✓ Labeled as Earthquake")
                break
            elif response == '1':
                labels[filename] = 1
                print("✓ Labeled as Quarry Blast")
                break
            else:
                print("Invalid input. Enter 0, 1, s, or q")
    
    save_labels(labels)


def save_labels(labels):
    """
    Save labels to JSON file
    """
    if len(labels) == 0:
        print("No labels to save!")
        return
    
    with open('labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n✓ Saved {len(labels)} labels to labels.json")
    print(f"  Earthquakes: {list(labels.values()).count(0)}")
    print(f"  Quarry Blasts: {list(labels.values()).count(1)}")


def create_labels_from_pattern():
    """
    Create labels based on filename patterns
    Useful if your files follow a naming convention
    """
    print("Scanning data directory...")
    files = scan_data_directory()
    
    if len(files) == 0:
        print("No HDF5 files found in data/ directory!")
        return
    
    print(f"Found {len(files)} HDF5 files\n")
    print("Enter pattern matching rules:")
    print("Example: Files containing '005' through '010' are earthquakes")
    print("         Files containing '011' through '020' are quarry blasts")
    
    earthquake_pattern = input("\nEnter substring for EARTHQUAKES (or press Enter to skip): ").strip()
    quarryblast_pattern = input("Enter substring for QUARRY BLASTS (or press Enter to skip): ").strip()
    
    labels = {}
    
    for filename in files:
        if earthquake_pattern and earthquake_pattern in filename:
            labels[filename] = 0
            print(f"✓ {filename} -> Earthquake")
        elif quarryblast_pattern and quarryblast_pattern in filename:
            labels[filename] = 1
            print(f"✓ {filename} -> Quarry Blast")
        else:
            print(f"? {filename} -> Not matched")
    
    if len(labels) > 0:
        save_labels(labels)
    else:
        print("\nNo files matched the patterns!")


def create_sample_labels():
    """
    Create a sample labels.json file for demonstration
    """
    files = scan_data_directory()
    
    if len(files) == 0:
        print("No HDF5 files found in data/ directory!")
        return
    
    # Create sample labels (first half = earthquakes, second half = quarry blasts)
    labels = {}
    mid = len(files) // 2
    
    for i, filename in enumerate(files):
        labels[filename] = 0 if i < mid else 1
    
    save_labels(labels)
    print("\n⚠ WARNING: This is a SAMPLE labeling!")
    print("   Please review and correct labels.json before training!")


def main():
    print("="*60)
    print("DAS Seismic Data Labeling Tool")
    print("="*60)
    print("\nOptions:")
    print("  1. Interactive labeling (label each file manually)")
    print("  2. Pattern-based labeling (use filename patterns)")
    print("  3. Create sample labels (for testing only)")
    print("  4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            create_labels_interactively()
            break
        elif choice == '2':
            create_labels_from_pattern()
            break
        elif choice == '3':
            create_sample_labels()
            break
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4")


if __name__ == '__main__':
    main()
