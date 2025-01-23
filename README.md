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

Once the input files are ready, you can run PPseq. This process is computationally intensive and should ideally be executed on an HPC system.

#### Steps:
1. **Create a new conda environment.**
2. **Activate the environment and clone the** [PPseq repository](https://github.com/EmmettJT/sequences_PPseq):


   ```bash
   git clone https://github.com/EmmettJT/sequences_PPseq/tree/emmett
   ```
   **Note:** This is a forked version of a private repository, so you will need to request access.
   
3. **Clone the submodule:**  
   ```bash
   git submodule update --init --recursive
   ```

4. **Switch submodule to branch "sacredSeqBranch"**

   ```bash
   cd PPSeq.jl
   git checkout sacredSeqBranch
   ```

---

### Awake Runs

Awake timeframes (during behavior) should be processed using the Julia script `PPSeq_awake_emmett.jl`.

1. **Edit the Julia script:**  
   Open `PPSeq_awake_emmett.jl` in a text editor and update the `list_of_animals` variable. This should include all recordings (in the format `animalID_implant_recording`) for which you have prepared data.
   
2. **Run the script directly:**  
   ```bash
   julia PPSeq_awake_emmett.jl --data-directory <PATH_TO_PREPARED_DATA> \
                               --num-threads <NUM_THREADS> \
                               --results-directory <PATH_TO_OUTPUT> \
                               --slurm-array-task-id <INDEX_FOR_LIST_OF_ANIMALS>
   ```
   - Replace `<PATH_TO_PREPARED_DATA>`, `<NUM_THREADS>`, `<PATH_TO_OUTPUT>`, and `<INDEX_FOR_LIST_OF_ANIMALS>` with the appropriate values.

3. **Or use the provided SLURM batch file:**  
   - Update the paths and modify the `#SBATCH` settings in `batch_awake_emmett` (e.g., adjust `--array=0-[NUMBER_OF_RECORDINGS_TO_RUN]`).
   - Execute the SLURM file:  
     ```bash
     sbatch batch_awake_emmett
     ```
   - Use the command `squeue` to monitor job progress.

---

### Sleep Runs

Sleep timeframes should be processed using the Julia script `PPSeq_sleep_emmett.jl`.

1. **Edit the Julia script:**  
   Open `PPSeq_sleep_emmett.jl` in a text editor and update the `list_of_animals` variable. This should include all recordings (in the format `animalID_implant_recording`) for which you have prepared data.
   
2. **Run the script directly:**  
   ```bash
   julia PPSeq_sleep_emmett.jl --data-directory <PATH_TO_PREPARED_DATA> \
                               --num-threads <NUM_THREADS> \
                               --results-directory <PATH_TO_OUTPUT> \
                               --number-of-sequence-types <NUM_SEQUENCE_TYPES> \
                               --sacred-directory <PATH_TO_AWAKE_PPSEQ_OUTPUT> \
                               --slurm-array-task-id <INDEX_FOR_LIST_OF_ANIMALS>
   ```
   - Replace `<PATH_TO_PREPARED_DATA>`, `<NUM_THREADS>`, `<PATH_TO_OUTPUT>`, `<NUM_SEQUENCE_TYPES>`, `<PATH_TO_AWAKE_PPSEQ_OUTPUT>`, and `<INDEX_FOR_LIST_OF_ANIMALS>` with the appropriate values.
   - **Key flags:**
     - `--sacred-directory`: Points PPseq to the Awake output folder to use parameters determined during Awake training for sequence search in sleep.
     - `--number-of-sequence-types`: Specifies how many sequences to fit. The default is 6. For sleep runs, it's recommended to use the number from Awake runs plus 2 (to account for non-task-related activity).

3. **Or use the provided SLURM batch file:**  
   - Update the paths, flags, and modify the `#SBATCH` settings in `batch_sleep_emmett` (e.g., adjust `--array=0-[NUMBER_OF_RECORDINGS_TO_RUN]`).
   - Execute the SLURM file:  
     ```bash
     sbatch batch_sleep_emmett
     ```
   - Use the command `squeue` to monitor job progress.

---

For more informaiton on running PPseq on the clsuter and changing PPseq paramters please refer to the following:

1. README of the [PPseq repository](https://github.com/EmmettJT/sequences_PPseq) 
2. The Methods section of our publication [Replay of Procedural Experience is Independent of the Hippocampus](https://www.biorxiv.org/content/10.1101/2024.06.05.597547v1.full.pdf) 
3. The [origional ppseq publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC8734964/) 

**Note:**  
PPseq is computationally intensive and may take several hours or even days to complete, depending on the dataset size.
















