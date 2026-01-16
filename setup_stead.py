"""
Alternative STEAD setup using SeisBench library
SeisBench provides programmatic access to STEAD without manual download
"""

def download_stead_with_seisbench():
    """
    Download STEAD dataset using SeisBench library
    This is easier than manual download but requires internet connection
    """
    try:
        import seisbench.data as sbd
        from pathlib import Path
        import pandas as pd
        import h5py
        
        print("="*70)
        print("Downloading STEAD using SeisBench")
        print("="*70)
        print("\nThis may take a while depending on your internet connection...")
        print("SeisBench will cache the data locally.\n")
        
        # Load STEAD dataset
        stead = sbd.STEAD()
        
        print(f"\nSTEAD dataset loaded:")
        print(f"  Total waveforms: {len(stead)}")
        print(f"  Metadata columns: {list(stead.metadata.columns)}")
        
        # Save metadata to CSV for compatibility with our loader
        output_dir = Path("data/stead")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "stead_seisbench_metadata.csv"
        stead.metadata.to_csv(csv_path, index=False)
        print(f"\n✓ Saved metadata to: {csv_path}")
        
        print("\n" + "="*70)
        print("SeisBench Setup Complete!")
        print("="*70)
        print("\nNote: SeisBench uses its own data format.")
        print("You'll need to adapt the loader to use SeisBench's API.")
        print("\nExample usage:")
        print("  from seisbench.data import STEAD")
        print("  stead = STEAD()")
        print("  waveforms, metadata = stead.get_sample(0)")
        print("\nSee: https://github.com/seisbench/seisbench")
        
        return True
        
    except ImportError:
        print("\n" + "="*70)
        print("ERROR: SeisBench not installed")
        print("="*70)
        print("\nInstall SeisBench:")
        print("  pip install seisbench")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\nError downloading STEAD: {e}")
        return False


def verify_stead_manual():
    """
    Verify manually downloaded STEAD dataset
    """
    from pathlib import Path
    import h5py
    import pandas as pd
    
    print("="*70)
    print("Verifying STEAD dataset")
    print("="*70)
    
    stead_dir = Path("data/stead")
    hdf5_path = stead_dir / "merge.hdf5"
    csv_path = stead_dir / "merge.csv"
    
    issues = []
    
    # Check if files exist
    if not hdf5_path.exists():
        issues.append(f"❌ Missing: {hdf5_path}")
    else:
        print(f"✓ Found: {hdf5_path}")
        file_size_gb = hdf5_path.stat().st_size / (1024**3)
        print(f"  Size: {file_size_gb:.2f} GB")
        
        # Check HDF5 file integrity
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'data' in f:
                    num_traces = len(f['data'].keys())
                    print(f"  Traces: {num_traces:,}")
                else:
                    issues.append("❌ No 'data' group in HDF5 file")
        except Exception as e:
            issues.append(f"❌ Cannot open HDF5 file: {e}")
    
    if not csv_path.exists():
        issues.append(f"❌ Missing: {csv_path}")
    else:
        print(f"✓ Found: {csv_path}")
        
        # Check CSV file
        try:
            df = pd.read_csv(csv_path)
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            # Check required columns
            required_cols = ['trace_name', 'trace_category']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues.append(f"❌ Missing columns in CSV: {missing_cols}")
            else:
                print(f"  ✓ All required columns present")
                
                # Show distribution
                print(f"\nTrace categories:")
                for cat, count in df['trace_category'].value_counts().items():
                    print(f"    {cat}: {count:,}")
                    
        except Exception as e:
            issues.append(f"❌ Cannot read CSV file: {e}")
    
    # Summary
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✓ STEAD dataset verified successfully!")
        print("="*70)
        print("\nYou can now run:")
        print("  python train_stead_test_das.py")
        return True
    else:
        print("⚠ Issues found:")
        print("="*70)
        for issue in issues:
            print(issue)
        print("\nPlease download STEAD dataset:")
        print("  bash setup_stead.sh")
        print("or see README_STEAD.md for manual download instructions")
        return False


def main():
    import sys
    
    print("\n" + "="*70)
    print("STEAD Dataset Setup Tool")
    print("="*70)
    print("\nOptions:")
    print("  1. Download using SeisBench (automatic)")
    print("  2. Verify manually downloaded dataset")
    print("  3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n")
        success = download_stead_with_seisbench()
        if success:
            print("\n✓ Success!")
        else:
            print("\n❌ Failed. Try manual download:")
            print("   bash setup_stead.sh")
            
    elif choice == '2':
        print("\n")
        verify_stead_manual()
        
    else:
        print("\nExiting...")
        print("\nTo download STEAD manually:")
        print("  bash setup_stead.sh")
        print("\nOr download directly from:")
        print("  https://rebrand.ly/whole")


if __name__ == '__main__':
    main()
