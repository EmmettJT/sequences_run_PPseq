# Running PPseq

Processing pipeline for preparing data for a PPseq run and executing a PPseq run for the sequences project.  
- For more details, see our publication: [Replay of Procedural Experience is Independent of the Hippocampus](https://www.biorxiv.org/content/10.1101/2024.06.05.597547v1.full.pdf)  

*Author: Emmett J. Thompson*

---

## Overview

This repository aims to:
1. Process data and determine an appropriate timing range for running PPseq.
2. Generate the input files required for a PPseq run.
3. Execute a PPseq run in either "awake" or "replay" mode.  
   *(Note: To run PPseq in replay mode, the data must first be processed in awake mode to train the model.)*

**Prerequisite:**  
This pipeline assumes data has been preprocessed using the following repository:  
https://github.com/EmmettJT/sequences_neuropixel_preprocess  
The scripts expect specific files and a predefined directory structure.

---

## Directory Structure

Organized data should follow this structure, containing folders for animals and recordings:

```plaintext
organised_data
    └── animals
        ├── [animal_1]
        │   ├── [recording_1]: behav_sync, ephys, video
        │   ├── [recording_2]: behav_sync, ephys, video
        │   └── ...
        ├── [animal_2]
        │   ├── [recording_1]: behav_sync, ephys, video
        │   ├── [recording_2]: behav_sync, ephys, video
        │   └── ...
        └── ...
```

### Key Folders:

1. **`behav_sync`:**  
   Contains folders for each experimental session with data aligning ephys, video, and behavioral timestamps.

2. **`ephys`:**  
   - Preprocessed data for each probe, including Kilosort output.  
   - Contains "good" and "MUA" spike cluster files, which include unit ID, spike times, unit depth, and region (e.g., striatum, motor cortex).

3. **`video`:**  
   - Raw video files with uncycled timestamp/trigger time dataframes.  
   - Tracking data for each video and tracking type.

**Note:**  
While paths in the notebooks can be modified, the scripts assume a specific data format. Refer to the preprocessing repo for details on the required file contents.

---

## Pipeline Guide

### 1. Performance Assessment for PPseq

This step calculates task performance for the given animal based on the behavior synchronization file.

#### Outputs:
- **Performance score**: Saved as a `.csv` file and plotted as a `.png` in a new `post_process_ppseq` folder.  
- **Time_intervals.txt**: Used to define the ephys timeframe for PPseq.  

#### Aim:  
Identify an ephys timeframe when the animal performed the task well/consistently. It’s recommended to finalize this decision after running Step 3 once for better context.

---

### 2. Identifying Sleep Periods

This step uses sleep tracking and synchronization files to identify sleep periods.

#### Outputs:
- **Average movement velocity**: Saved as `.csv` files in the `post_process_ppseq` folder.  

#### Aim:  
Determine sleep time periods in *Time_intervals.txt* based on low movement velocity. It’s recommended to finalize this decision after running Step 3 once for better context.

---

### 3. Preparing Data for PPseq

This step prepares data for PPseq runs. Run this step twice:  
1. Initial run to generate plots for chosing a time interval.  
2. Use the plots to refine the time range and run again.

#### Key Parameters:
- **`Time_span`**: Defines the timeframe (e.g., "Awake" or "PostSleep"). Multiple timeframes can be defined as lists of intervals (e.g., `[[time1, time2], [time3, time4]]`).  
- **`region`**: Specifies the brain region of interest based on spike data.  
- **Other parameters**:  
  - `min_fano_factor` / `max_fano_factor`: Filter neurons based on the variance-to-mean spike ratio.  
  - `max_firing_rate`: Excludes neurons with excessively high firing rates.  
  - `single_or_multiunits`: Options are "good," "MUA," or "both" (recommended).  
  - `shuffle`: Shuffles neuron IDs (for testing PPseq).  
  - `visualise`: If `True`, generates diagnostic plots (recommended but slower).

#### Outputs:
- **Prepared data**: Parameters JSON and spikes file saved in a "prepared data" folder.  
- **Plots**: Visualizations of time ranges, firing rate, movement velocity, trial occurrence, and task performance.

#### Aim:  
Refine the time range and produce input files for PPseq.

---

### 4. Running PPseq

Once the input files are ready, run PPseq. This is computationally intensive and should ideally be done on an HPC system.

#### Steps:
1. Create a new conda environment
2. Activate envionrment and clone the PPseq repository:  
   https://github.com/EmmettJT/sequences_PPseq/tree/emmett
   **note** this is a forked version of a private repo and so you will need to request access. 
3. Clone the submodule:  
   ```bash
   git submodule update --init --recursive

#### Awake runs
Awake timeframes (during behaviour) should be run by calling the julia file  

**Note:**  
PPseq may take hours or even days to complete, depending on the dataset size.



























