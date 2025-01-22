# Running PPseq
Processing pipeline for prepareing data for a PPseq run and executing a PPseq run for the sequences project 
- See our publication for more details: [Replay of Procedural Experience is Independent of the Hippocampus](https://www.biorxiv.org/content/10.1101/2024.06.05.597547v1.full.pdf)
  
Emmett J Thompson


# Aim
The aim of this repo is as follows
1. Take processed data use it to decide on a timing range in which to run ppseq, 
2. Produce the input files PPseq needs in order to run
3. Execute a run of PPseq


# Steps
1. 
2. ff
3. ff
4. run prepare data script to generate PPseq input files
5. f
6. f
7. 



Clone the PPseq repo:
https://github.com/EmmettJT/sequences_PPseq/tree/emmett
clone in the submodule PPseq:
git submodule update --init --recursive


Make sure you are on the branch 'emmett'










# Important:
This pipeline assumes data has been preprocessed using the following repo: https://github.com/EmmettJT/sequences_neuropixel_preprocess
The scripts are expecting certain files and a certain file structure to exist already. 

# file structure: 

an organised data file containing animal files which themselves containt a file for each recoridng for that animal. Each recording contains the same information: ephys, video and syncronisation files

to see an example of this structure for yourself go to 'ceph\projects\sequence_squad\revision_data\organised_data\'

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

the final shared files across all recordings: 

#### behav_sync
- contains folders for each experimetnal section. these folders contain dataframes which align the three experimental clocks - ephys, video and behavioural data.  
#### ephys
- contains preprocessed data for each probe including kiolosort output
- importnalty, for these scripts this folder contains good and mua spike clusters - dataframes with unit id, spikes times, unit depth and the region this unit was found in (eg. striatum, m_cortrex...) 
#### video 
- contains a file with the raw video files (wit their uncycled timestamp/trigger time dataframes)
- contains a file with tracking data for each video (and each tracking type) 

**The scripts and paths in each notebook file should be reasonably easy to alter, in order to use a different file structure. However, the scripts also expect a certain format for the loaded data files themselves, see the preprocessing repo mentione dabove to understand how these files are created. 



