# seis-learning-spatial: Application of human engineered features for Fiber-Optics Distributed Acoustic Sensing records
***
This repository contains a collection of scripts designed for training and evaluating a machine learning model using Distributed Acoustic Sensing (DAS) data. The machine learning algorithm leverages a set of 111 features detailed in the associated scientific publication:
***C. Huynh, Hibert C., Jestin C., Malet J.-P., Lanticq V. 2024. A real-scale application of a novel set of spatial and similarity features for detection and classification of natural seismic sources from Distributed Acoustic Sensing data. Geophysical Journal International (GJI).***

The dataset required for this repository is available online *(DAS-BIGORRE_(date_time).h5)* and should be placed in the **data** directory of this project. To skip the feature computation step, you can download in addition the file *(data_tables_8-1000_bloc.h5)* , place in the **features** directory and use it directly with the [E_crossval.py](E_crossval.py) script.

***

## Scripts description
This repository contains a set of 5 scripts intended to be run sequentially:
- [A_compute_eb.py](A_compute_eb.py): Script to compute Energy Bands (EB) from Strain Rate data (SR) stored in HDF5 files. The script checks if the specified EB already exists; if not, it computes and stores the EB in the HDF5 file.
- [B_compute_labelization.py](B_compute_labelization.py): This script allows users to interactively label regions of an Energy Band (EB) plot. Users can click on the plot to draw rectangles and assign labels to them. The labeling process involves three clicks: the first two define the corners of the rectangle, and the third click selects a label from the list displayed on the right side of the plot.
- [C_compute_features.py](C_compute_features.py): This script processes strain rate seismic data to compute features. It manages large datasets by dividing them into smaller subparts, performs feature computation, and saves the results into an HDF5 file.
- [D_merge_features.py](D_merge_features.py): Merge the attributes calculated for each event file into a single file containing pandas dataframes: "file_df", "event_df", "reference_df", "dwindow_df", "twindow_df", "features_df". Ready to be used for machine learning process.
- [E_crossval.py](E_crossval.py): Script to perform Leave One Out Cross-Validation (LOOCV) of XGBoost models trained on provided features, with thresholding on score for prediction.


## Input and output data
The script [A_compute_eb.py](A_compute_eb.py) needs a specific DAS data input. Input data needs to be HDF5 file with one main group named "python_processing". Subgroup of "python_processing" should be the following:

- **sr** : strain rate, it corresponds to raw seismic data
    - **data**  
        *Content*:  amplitude of sr  
        *Data type*: 2d array, shape: (nb_channel, temporal_pt_sr)  
    - **distance**  
        *Content*: position (in meters) associated to each sr location  
        *Data type*: 1d array, shape: (nb_channel,)  
    - **time**  
        *Content*: temporal sampling (in seconds)  
        *Data type*: 1d array, shape: (temporal_pt_sr,)  	

At the end of the [B_compute_labelization.py](B_compute_labelization.py) script, two subgroups are appended to the initial file:

- **eb** : energy band, it corresponds to energy integration  
    - **0.5_100** : frequency band choice  
        -  **data**  
    		*Content*:  amplitude of eb  
            *Data type*: 2d array, shape: (nb_channel, temporal_pt_eb)  
        - **distance**  
            *Content*: position (in meters) associated to each eb location  
            *Data type*: 1d array, shape: (nb_channel,)  
        - **time**   
            *Content*: temporal sampling (in seconds)  
            *Data type*: 1d array, shape: (temporal_pt_eb,)  

- **region** : region map. The label map can be deduced from this map  
    - **data**   
        *Content*: region map, area with the same integer belong to the same region  
        *Data type*: 2d array, shape: (nb_channel, temporal_pt_region)  
    - **distance**   
		*Content*: position (in meters) associated to each region location  
        *Data type*: 1d array, shape: (nb_channel,)  
    - **time**   
		*Content*: temporal sampling (in seconds)  
        *Data type*: 1d array, shape: (temporal_pt_region,)  
    - **region_id**   
        *Content*: list of integer representing the individual region. The i-th element of region_id is associated with the i-th element of classe_str   
        *Data type*: 1d array, shape: (nb_region,)  
    - **classe_str**   
        *Content*: list of label for each region  
        *Data type*: 1d array, shape: (nb_region,)  

where:
- **nb_channel**: number of channels (ie virtual sensor) along the fiber
- **temporal_pt_sr**: number of points in time direction for sr data
- **temporal_pt_eb**: number of points in time direction for eb data
- **temporal_pt_region**: number of points in time direction for region data
- **nb_region**: number of different region in the data


The script [D_merge_features.py](D_merge_features.py) generates a pandas-compatible HDF5 file organized as follows:

- **label_df** : dataframe associating id (int) and name (str) of the class. (pd.DataFrame)
- **file_df** : dataframe associating id (int) and name (str) of the file. (pd.DataFrame)
- **event_df** : dataframe associating event id (int), file id (int), and class id (int). (pd.DataFrame)
- **reference_df** : dataframe associating signal id (int), file id (int), event id (int), window id (int), channel id (int), and class id (int). (pd.DataFrame)
- **channel_df** : dataframe associating channel id (int), file id (int), and location by distance in this file (int). (pd.DataFrame)
- **window_df** : dataframe associating window id (int), file id (int), and location by time in this file (int). (pd.DataFrame)
- **features_df** : dataframe associating signal id (int) with the different calculated values for the features (float). (pd.DataFrame)

