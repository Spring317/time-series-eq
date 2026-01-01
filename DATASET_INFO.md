# DAS-BIGORRE-2022 Dataset Information

## Dataset Overview

- **Name**: DAS-BIGORRE-2022
- **Location**: Bigorre, Hautes-Pyrénées, France
- **Period**: August 31 - September 20, 2022
- **Cable Length**: 91 km fiber optics
- **Total Events**: 19 (10-minute records each)

## Event Distribution

| Category | Count | Label |
|----------|-------|-------|
| **Earthquakes** | 13 | 0 |
| **Quarry Blasts** | 6 | 1 |

## Files with Labels

### Earthquakes (13 events)
1. `DAS-BIGORRE_2022-09-03_03-46-50_UTC-010.h5` - Mw 0.6
2. `DAS-BIGORRE_2022-09-03_13-06-50_UTC-013.h5` - Mw 1.0
3. `DAS-BIGORRE_2022-09-03_18-26-50_UTC-018.h5` - Mw 2.4
4. `DAS-BIGORRE_2022-09-05_00-46-50_UTC-007.h5` - Mw 1.4
5. `DAS-BIGORRE_2022-09-06_07-56-50_UTC-003.h5` - Mw 1.2
6. `DAS-BIGORRE_2022-09-08_17-06-50_UTC-004.h5` - Mw 0.8
7. `DAS-BIGORRE_2022-09-09_07-06-50_UTC-017.h5` - Mw 2.0
8. `DAS-BIGORRE_2022-09-09_17-36-50_UTC-012.h5` - Mw 0.4
9. `DAS-BIGORRE_2022-09-15_09-06-50_UTC-009.h5` - Mw 1.1
10. `DAS-BIGORRE_2022-09-16_11-16-50_UTC-019.h5` - Mw 1.6
11. `DAS-BIGORRE_2022-09-16_23-06-50_UTC-008.h5` - Mw 1.1
12. `DAS-BIGORRE_2022-09-18_04-16-50_UTC-011.h5` - Mw 0.8
13. `DAS-BIGORRE_2022-09-20_04-36-50_UTC-015.h5` - Mw 1.3

### Quarry Blasts (6 events)
1. `DAS-BIGORRE_2022-08-31_07-59-21_UTC-005.h5` - Mw 0.7
2. `DAS-BIGORRE_2022-09-01_09-19-21_UTC-016.h5` - Mw 0.8
3. `DAS-BIGORRE_2022-09-06_10-06-50_UTC-020.h5` - Mw 1.0
4. `DAS-BIGORRE_2022-09-08_09-56-50_UTC-001.h5` - Mw 1.1
5. `DAS-BIGORRE_2022-09-12_10-06-50_UTC-002.h5` - Mw 0.6
6. `DAS-BIGORRE_2022-09-14_09-16-50_UTC-014.h5` - Mw 1.1

## HDF5 Data Structure

Each file contains:

### `sr` (Strain Rate - Raw Seismic Data)
- **data**: (nb_channels, time_samples) - Amplitude of strain rate
- **distance**: (nb_channels,) - Position in meters for each channel
- **time**: (time_samples,) - Temporal sampling in seconds

### `eb` (Energy Band)
- **0.5_100/data**: Energy integration data
- **0.5_100/distance**: Position data
- **0.5_100/time**: Time data

### `region` (Region Map)
- **data**: Region segmentation map
- **region_id**: Integer IDs for each region
- **classe_str**: Label strings for each region

## Data Characteristics

- **Channels**: ~9000 (91 km cable with ~10m spacing)
- **Sampling Rate**: 200 Hz
- **Duration**: 10 minutes per file
- **Format**: HDF5
- **Data Type**: Strain rate measurements

## Label Generation

Labels are automatically generated from the dataset documentation using:
```bash
python generate_labels.py
```

This creates:
- `labels.json` - Numeric labels (0=EQ, 1=QB)
- `dataset_metadata.json` - Detailed metadata with magnitudes

## Training Pipeline

The training pipeline:
1. Reads `sr/data` from HDF5 files (strain rate data)
2. Uses sliding windows (2000 samples = 10s at 200Hz)
3. Processes all ~9000 channels per window
4. Applies per-channel normalization
5. Classifies as Earthquake (0) or Quarry Blast (1)

## Memory Management

With ~9000 channels and 120,000 time samples (10 min at 200Hz):
- Full file in memory: ~8.6 GB (float64)
- Windowed approach: ~172 MB per window (2000 samples)
- Batch of 16 windows: ~2.7 GB

The pipeline uses lazy loading to handle this efficiently with 32GB RAM.

## References

**Dataset Publication:**
C. Huynh, Hibert C., Jestin C., Malet J.-P., Lanticq V. 2024. 
*A real scale application of a novel set of spatial and similarity features for detection and classification of natural seismic sources from Distributed Acoustic Sensing data.* 
Geophysical Journal International (GJI).

**Event Catalog:**
BCSF-RENASS service: https://renass.unistra.fr/
