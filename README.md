# Running PPseq
Processing pipeline for prepareing data for a PPseq run and executing a PPseq run for the sequences project 
- See our publication for more details: [Replay of Procedural Experience is Independent of the Hippocampus](https://www.biorxiv.org/content/10.1101/2024.06.05.597547v1.full.pdf)
  
Emmett J Thompson

----
# IMPORTANT INFO
----
## What does this repo do?
The aim of this repo is as follows
1. Take processed data use it to decide on a timing range in which to run ppseq, 
2. Produce the input files PPseq needs in order to run
3. Execute a run of PPseq in awake or replay mode (To run data with Ppseq in replay mode, the data first has to be PPseq'ed in Awake mode to train the model) 

**This pipeline assumes data has been preprocessed using the following repo: https://github.com/EmmettJT/sequences_neuropixel_preprocess
The scripts are expecting certain files and a certain file structure to exist already. 

## File structure: 
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
----
## The final shared files across all recordings: 
#### behav_sync
- contains folders for each experimetnal section. these folders contain dataframes which align the three experimental clocks - ephys, video and behavioural data.
#### ephys
- contains preprocessed data for each probe including kiolosort output
- importnalty, for these scripts this folder contains good and mua spike clusters - dataframes with unit id, spikes times, unit depth and the region this unit was found in (eg. striatum, m_cortrex...)
#### video 
- contains a file with the raw video files (wit their uncycled timestamp/trigger time dataframes)
- contains a file with tracking data for each video (and each tracking type)

**The scripts and paths in each notebook file should be reasonably easy to alter, in order to use a different file structure. However, the scripts also expect a certain format for the loaded data files themselves. This may make the scirpts difficult to adapt but understanding the logic of each step should mean that you can adapt the scripts for your data - please see the preprocessing repo mentioned above to understand what data each file contains. 



----


# PROCESSING GUIDE

## 1. Performance for PPseq
- This script loops across the data structure for the given animal and uses the behaviour synchronisation file which relates preprocessed bpod data (poke times) to ephys timestamps. 
#### Output 
- A performance score (how well the mouse did the sequence across trials) in ephys time coordinates, saved out as a '.csv' file (and plotted as a '.png') in a new folder in the recording directory called 'post_process_ppseq'
- This script also creates a file called 'Time_intervales.txt.'. This file is used to decide which ephys timeframe to feed into PPseq. Currently it will be set to an arbitrary time range.
#### Aim
- This data allows us to chose an ephys time period when the animal was perfroming the task well/consitently. This can be done now based on the saved out performance plot, however, I reccomend that you wait until after running prepare data (step 3) once, as this will give you more information to make this decision - although at no point is this decision 'final'. it can be altered freely (see step 3) 

----

## 2. Find sleeping range
- This script loops across the data structure (all recordings) for a given animal and uses the sleep tracking files along with the synchronisation files.
#### Output
- a average moevment velocity for pre and postsleep periods, saved out as '.csv' files in the 'post_process_ppseq' folder
#### Aim

----

## 3. Prepare data (for ppseq)
- This script takes in a list of recordings (in the format animal_implant_recording) and produces prepared data for a PPseq run

#### Parameters
The most impoartant parameters are *Time_span* and *region*.
Time_span: eg. 'Awake' or 'PostSleep' refers to the timespan defined in 'Time_intervales.txt.'. You can define mulitple timespans (not reccomended for Awake training), these should be set as a list of lists. eg. two Awake timeframes 1-2 and 3-4 = 'Awake,[[time1,time2],[time3,time4]]'
region: this paramter refers to the brain region of interest. The spike dataframes which are fed into this script should contain a column defineing their location in the brain. This paramter defes which units get excluded or inlcuded based on this information

Other paramterd: 
min_fano_factor & max_fano_factor - these filter the data based on the ratio of the variance to the mean number of spikes in a neuron
max_firing_rate - removes neurons with a very high firing rate (Ppseq fits to spikes so tonic interneurons with huge numbers of spikes will dominate the model)
single_or_multiunits - use units kilosort has defined as good or mua (reccomeded to use 'both') 
shuffle - if True, shuffles the neuron IDs (for testing Ppseq) 
visualise = 'True' will produce plots (This slows things down but is recomended) 

#### Output
- params json and spikes file (from the defined time interval) in a prepared data folder
- Also produces a folder called 'plots' which shows the chosed time range in context with firing rate, movement velocity, trial occurance, and task performance. 

#### Aim


----

## 3. Run PPSeq
- This script takes in a list of recordings (in the format animal_implant_recording) and produces prepared data for a PPseq run 
#### Awake mode
- a average moevment velocity for pre and postsleep periods, saved out as '.csv' files in the 'post_process_ppseq' folder
#### Replay mode


Clone the PPseq repo:
https://github.com/EmmettJT/sequences_PPseq/tree/emmett
clone in the submodule PPseq:
git submodule update --init --recursive


Make sure you are on the branch 'emmett'












