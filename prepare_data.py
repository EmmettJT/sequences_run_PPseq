import os.path
# from typing import Annotated
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import argparse
import json

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-filename', type=str, default="defaultData",
        help='output filename')
    parser.add_argument(
        '--save-path', type=str, default="./data/preparedData/",
        help='save directory, it creates one if the path doesnt exist')
    parser.add_argument(
        '--input-data-path', type=str, default="./data/rawData/")
    parser.add_argument(
        '--number-of-neurons', type=int, default=215,
        help='number of neurons')
    parser.add_argument(
        '--time-span', type=str, default="[[1500, 2000]]",
        help='List of lists with time spam to be concatenated')
    parser.add_argument(
        '--use-curated', type=str, default="True",
        help='if True, use the curated spike trains, if False, use the whole thing')
    parser.add_argument(
        '--use-behaviourally-curated', type=str, default="False",
        help='if True, use the curation list emmett made which chooses neurons which aline well with behavioural data (this didnt work very well in past)')
    parser.add_argument(
        '--remove-top-N-neurons', type=int, default=0,
        help='if True, remove top N active neurons')
    parser.add_argument(
        '--align-to-zero', type=str, default="True",
        help='if True, start spike trains from 0')
    parser.add_argument(
        '--neuron-indices-file', type=str, default = None,
        help='If set points the program to a json file that it uses to cut down the right set of neurons. Just give the name of the file after "./.../params_"')
    parser.add_argument(
        '--shuffle', default='None', type=str,
        help="options for shuffling data: strings: 'None', 'shuffle_clusters', 'jitter_timeToJitterInSecs' e.g. 'jitter_0.1' ")
    parser.add_argument(
        '--remove-high-firing-rate', type=float, default=1000.0,
        help="Remove neurons with firing rate higher than the one specified (in Hz)")
    parser.add_argument(
        '--min-fano-factor', type=float, default=0.0,
        help='Remove neurons with Inter-Spike Interval Fano Factors above this number')
    parser.add_argument(
        '--max-fano-factor', type=float, default=1e5,
        help='Remove neurons with Inter-Spike Interval Fano Factors below this number')

    args = parser.parse_args(argv)
    args = vars(args)
    args["time_span"] = eval(args["time_span"])
    args["use_curated"] = eval(args["use_curated"])
    args["use_behaviourally_curated"] = eval(args["use_behaviourally_curated"])
    args["remove_top_N_neurons"] = args["remove_top_N_neurons"]
    args["align_to_zero"] = eval(args["align_to_zero"])
    args["remove_high_firing_rate"] = args["remove_high_firing_rate"]

    if not os.path.exists(args["save_path"]):
        os.makedirs(args["save_path"])
                
    print("\nPARSED ARGUMENTS: ")
    for key, value in args.items():
         print("   ","{:<26}".format(key), "{:<20}".format(str(type(value))), value)

    # print("\nLOADING DATA")
    sampleRate = 30000.0 #sample rate
    offset = 0 #recording offset 
    dataDirectory = args["input_data_path"] #where the data is stored

    #load raw data files
    print("    loading raw data files...")
    discreteSpikeTimes = (np.concatenate(np.load(dataDirectory + 'spike_times.npy')))/sampleRate #+ offset OFFSET ALREADY ACCOUNTED FOR BY EMMETT
    clusterIdentities = np.load(dataDirectory + 'spike_clusters.npy') #same length as discreteSpikeTimes
    mScores = pd.read_csv(dataDirectory + 'cluster_group.tsv', sep='\t')
    print(f"      {discreteSpikeTimes.shape[0]} total spikes")

    print("\nPREPARING DATA")
    #converts spike data from long list of spike times into many lists of spike times, one for each cluster
    print("    converting to list of lists...")
    discreteSpikeTimes, clusterIDs = splitByCluster(discreteSpikeTimes,clusterIdentities)
    print(f"      {clusterIDs.shape} neurons")

    #removes clusters which aren't manually annotated as 'good'
    if args["use_curated"] == True: 
        print("    removing clusters which aren't manually annotated as 'good'...")
        manualScore = pd.read_csv(dataDirectory + 'cluster_group.tsv', sep='\t')
        discreteSpikeTimes, clusterIDs = filterByManualCurationScore(discreteSpikeTimes,clusterIDs,manualScore)
        print(f"      {clusterIDs.shape} neurons")

    #removes clusters which aren't on behavioural curation list 
    if args["use_behaviourally_curated"] == True:
        print("    removing clusters which aren't on behavioural curation list...")
        behaviourScore = pd.read_csv(dataDirectory + 'manually_curated_clusters.csv')
        discreteSpikeTimes, clusterIDs = filterByBehaviouralCurationScore(discreteSpikeTimes,clusterIDs,behaviourScore)
        print(f"      {clusterIDs.shape} neurons")

    # If you've given a neurons_to_keep file use that
    if args["neuron_indices_file"] != None:
        print("     cropping neurons according to provided json ...")
        cropped_neuron_file = args["save_path"] + "params_" + args["neuron_indices_file"] + ".json"
        cropping_file = open(cropped_neuron_file,)
        parameters = json.load(cropping_file)
        cropped_neuron_IDs = parameters['neuronIDs']

        discreteSpikeTimes, clusterIDs = filterByNeuronList(discreteSpikeTimes, clusterIDs, cropped_neuron_IDs)
        print(f"        {clusterIDs.shape} nuerons")
    
    print("    cropping to desired time spans...")
    timeSlices = []
    discreteSpikeTimes = [np.array(dst) for dst in discreteSpikeTimes] #make into list of arrays not list of lists 
    concatenation_shift = 0
    for timeSpan in args["time_span"]:
        timeSlice = [el[(el>timeSpan[0]) & (el<timeSpan[1])] for el in discreteSpikeTimes]
        if args["align_to_zero"]:
            shiftby = timeSpan[0]-concatenation_shift
            timeSlice = [el - shiftby for el in timeSlice]
            concatenation_shift += timeSpan[1] - timeSpan[0]
        timeSlices = timeSlices + [timeSlice]
    discreteSpikeTimes = []
    for i in range(len(timeSlices[0])):
        neuronSpikeTimes = []
        for j in range(len(timeSlices)):
            neuronSpikeTimes = neuronSpikeTimes + list(timeSlices[j][i])
        discreteSpikeTimes += [neuronSpikeTimes]
    print(f"      {sum([len(dst) for dst in discreteSpikeTimes])} total spikes")

    # First see if we should be removing based on fano factor
    if args["min_fano_factor"] > 0.0:
        print("    ordering neurons by ISI fano factor...")
        ISI_var = np.zeros([len(discreteSpikeTimes)])
        ISI_mean = np.zeros([len(discreteSpikeTimes)])
        ISI_fano = np.zeros([len(discreteSpikeTimes)])
        for (i, spikeList) in enumerate(discreteSpikeTimes):
            spike_times = np.array(spikeList)
            ISIs = spike_times[1:len(spike_times)] - spike_times[0:len(spike_times) - 1]
            ISI_var[i] = np.var(ISIs)
            ISI_mean[i] = np.mean(ISIs)
            ISI_fano[i] = ISI_var[i] / ISI_mean[i]
        idxs = np.flip(np.argsort(ISI_fano))

        # First check for NaNs, create an index for that
        num_nans = np.sum(np.isnan(ISI_fano))
        if num_nans > 0:
            print(f"    Removing {num_nans} neurons whose ISI fano is a NaN")
            nan_cut_index = np.logical_not(np.isnan(ISI_fano))

        # Then create an index for those with too low fano factor
        print(f"    removing lowest ISI fano factor neurons above {args['min_fano_factor']}")
        fano_cut_index = ISI_fano >= args['min_fano_factor']
        upper_fano_cut_index = ISI_fano <= args["max_fano_factor"]

        cut_index = np.logical_and(fano_cut_index, nan_cut_index)
        cut_index = np.logical_and(upper_fano_cut_index, cut_index)
        discreteSpikeTimes, clusterIDs = np.array(discreteSpikeTimes)[cut_index], np.array(clusterIDs)[cut_index]

    # Then check if we should see if there are neurons firing too fast, if so remove them. No neuron fires over 1000 Hz...
    if args["remove_high_firing_rate"] < 1000.0 :
        print("    ordering neurons by average activity level...")
        totalSpikeCounts = [len(spikeList) for spikeList in discreteSpikeTimes]
        idx = np.argsort(totalSpikeCounts)
        discreteSpikeTimes, clusterIDs = [discreteSpikeTimes[id] for id in idx], [clusterIDs[id] for id in idx]

        mostActive = len(discreteSpikeTimes[-1]) / (discreteSpikeTimes[-1][-1] - discreteSpikeTimes[-1][0])
        try:
            leastActive = len(discreteSpikeTimes[0]) / (discreteSpikeTimes[0][-1] - discreteSpikeTimes[0][0])
        except IndexError:
            leastActive = 0

        avg_firing_rate = []
        for spk in discreteSpikeTimes:
            try:
                rate = len(spk) / (spk[-1] - spk[0])
            except:
                rate = 0.0
            avg_firing_rate.append(rate)
        avg_firing_rate_mean = np.mean(avg_firing_rate)
        print(f"      most active neuron {mostActive :.2f}Hz, least active neuron {leastActive :.2f}Hz, Average {avg_firing_rate_mean:.2f}Hz")

        cut_index = np.array(avg_firing_rate) <= args["remove_high_firing_rate"]

        print("    removing most active neurons above %f Hz" % args['remove_high_firing_rate'])

        discreteSpikeTimes, clusterIDs = np.array(discreteSpikeTimes)[cut_index], np.array(clusterIDs)[cut_index]
        mostActive = len(discreteSpikeTimes[-1]) / (discreteSpikeTimes[-1][-1] - discreteSpikeTimes[-1][0])
        try:
            leastActive = len(discreteSpikeTimes[0]) / (discreteSpikeTimes[0][-1] - discreteSpikeTimes[0][0])
        except IndexError:
            leastActive = 0

        avg_firing_rate = []
        for spk in discreteSpikeTimes:
            try:
                rate = len(spk) / (spk[-1] - spk[0])
            except:
                rate = 0.0
            avg_firing_rate.append(rate)
        avg_firing_rate = np.mean(avg_firing_rate)
        print(f"      new average firing rate {avg_firing_rate :.2f}Hz")
        args["average_firing_rate"] = avg_firing_rate

        print(f"      new most active neuron {mostActive :.2f}Hz, least active neuron {leastActive :.2f}Hz")
        args["highest_freq_neuron"] = mostActive

    print("    removing most and least active neurons to get desired number...")
    if args["number_of_neurons"] > len(discreteSpikeTimes):
        print(f"      requested more neurons than are avaiable. Number of neurons remains at {len(discreteSpikeTimes)}")
    elif args["number_of_neurons"] == len(discreteSpikeTimes):
        print(f"      requested as many neurons as are avaiable. Number of neurons remains at {len(discreteSpikeTimes)}")
    elif args["number_of_neurons"] < len(discreteSpikeTimes):
        requestedNumber = args["number_of_neurons"]
        removeNumber = len(discreteSpikeTimes)-args["number_of_neurons"]
        removeLeastActive = int(np.round(removeNumber / 2))
        removeMostActive = removeNumber - removeLeastActive
        print(f"      requested {requestedNumber} neurons, whilst {len(discreteSpikeTimes)} are available. Removing the most active {removeMostActive} and least active {removeLeastActive} neurons")
        discreteSpikeTimes = discreteSpikeTimes[removeLeastActive:-removeMostActive]
        clusterIDs = clusterIDs[removeLeastActive:-removeMostActive]
    args["number_of_neurons"] = len(discreteSpikeTimes)

    if args["shuffle"] != 'None':
        print("    shuffling neurons to randomise data for control test, method: %s" %args["shuffle"])
        if args["shuffle"][:6] == 'jitter':
            timeToJitter = float(args["shuffle"][7:])
            print("        jittering by %.2f seconds" %timeToJitter)
            #addds random time to all spike times then reorders them 
            discreteSpikeTimes = [list(np.sort(np.array(times_list) + np.random.uniform(-timeToJitter,timeToJitter,size=(len(times_list))))) for times_list in discreteSpikeTimes]
        elif args["shuffle"] == 'shuffle_clusters':
            print("shuffling clusters")
            discreteSpikeTimes_concat = np.concatenate(discreteSpikeTimes)
            np.random.shuffle(discreteSpikeTimes_concat)
            new_discreteSpikeTimes = []
            k = 0
            for i in range(len(discreteSpikeTimes)):
                new_discreteSpikeTimes.append(list(np.sort(discreteSpikeTimes_concat[k:k+len(discreteSpikeTimes[i])])))
                k += len(discreteSpikeTimes[i])
            discreteSpikeTimes = new_discreteSpikeTimes
        else:
            print("unrecognised shuffle method")


    print("    re-ordering the cutdown neurons by absolute neuron index (from raw data)")
    sorted_clusterIDs = np.sort(clusterIDs)
    sorted_indices = np.argsort(clusterIDs)
    sorted_SpikeTimes = []
    for idx in range(len(sorted_clusterIDs)):
        sorted_SpikeTimes.append(discreteSpikeTimes[sorted_indices[idx]])
    
    

    
    print("\nSAVING DATA")
    args["neuronIDs"] = [int(id) for id in sorted_clusterIDs] 
    discreteSpikeTimes = [np.array(dst) for dst in sorted_SpikeTimes]

    data_filename = write_text_file(bound_spikes=discreteSpikeTimes, filename=os.path.join(args["save_path"], args["output_filename"]+".txt"))
    parameter_filename = save_params_as_json(args, args["save_path"], "params_"+args["output_filename"]+".json")
    print(f"   • data '.txt' file saved to:          {data_filename}")
    print(f"   • parameters '.json' file saved to:   {parameter_filename}")
    print("\n\n")


def splitByCluster(spikeTimes,clusters):
    """Takes 1D list of spike times and (same length) 1D list of cluster identities 
    and splits the times list into many lists of spike times, one for each cluster
    Args:
        spikeTimes (list): Spike timing lists
        clusters (list): Spike cluster DI list
    Returns:
        [list of lists]: Spike times split by cluster ID
    """    
    # creates spike time vectors for each unit and adjusts timestamps in line with start offset
    spikeTimeVectors = []
    clusterIDs = np.arange(1,max(clusters))
    for i in clusterIDs:
        spikeTimeVectors = spikeTimeVectors + [spikeTimes[np.where(clusters==i)]]
        # print(len(spikeTimeVectors[i-1]))
    return spikeTimeVectors, np.array(clusterIDs)

def filterByNeuronList(discreteSpikeTimes, clusterIDs, cropped_neurons):
    """Keeps a cluster if it appears in cropped_neurons, i.e. was used in another run
    Args:
        discreteSpikeTimes (list of lists ): spike time lists for every cluster
        clusterIDs (list of lists): cluster ID for each discrete spike time list 
        cropped_neurons: the list from the json of good neurons
    Returns:
        tuple: (list of lists: spike time lists for cropped neurons, list: the IDs of the cropped neurons)
    """

    croppedSpikeTimes = []
    croppedIDs = []

    for (index, idx) in enumerate(clusterIDs):
        if idx in cropped_neurons:
            croppedSpikeTimes = croppedSpikeTimes + [discreteSpikeTimes[index]]
            croppedIDs = croppedIDs + [idx]
    return (croppedSpikeTimes, np.array(croppedIDs))

def filterByManualCurationScore(discreteSpikeTimes,clusterIDs,manualScore):
    """Only keeps a cluster if its manual curation score is "good"
    Args:
        discreteSpikeTimes (list of lists ): spike time lists for every cluster
        clusterIDs (list of lists): cluster ID for each discrete spike time list 
        manualScore ([type]): pd dataframe for cluster ID and its manual annotation score
    Returns
        tuple: (list of lists: spike time lists for the "good" clusters, 
                list: the ID of the "good" clusters)
    """    
    curatedSpikeTimeseries = []
    curatedClusterIDs= []

    mScore = list(manualScore['group'])
    annotatedClusters = list(manualScore['cluster_id'])

    for (index,idx) in enumerate(clusterIDs):
        if (idx in annotatedClusters):
            if (mScore[annotatedClusters.index(idx)] == 'good'):
                curatedSpikeTimeseries = curatedSpikeTimeseries + [discreteSpikeTimes[index]]
                curatedClusterIDs = curatedClusterIDs + [idx]
    return (curatedSpikeTimeseries, np.array(curatedClusterIDs))


def filterByBehaviouralCurationScore(discreteSpikeTimes,clusterIDs,behaviouralScore,removeInterneurons=True):
    """Only keeps a cluster if its in the behavioural curation list
    """    
    curatedSpikeTimeseries = []
    curatedClusterIDs= []

    annotatedClusters = behaviouralScore['cluster id '].to_numpy()
    if removeInterneurons==True:
        annotatedClusters = annotatedClusters[np.where(behaviouralScore['Interneuron'].to_numpy() == 0)] #remove interneurons
    annotatedClusters = annotatedClusters[~np.isnan(annotatedClusters)]
    annotatedClusters = annotatedClusters.astype(int)


    
    for (index,idx) in enumerate(clusterIDs):
        if (idx in annotatedClusters):
            curatedSpikeTimeseries = curatedSpikeTimeseries + [discreteSpikeTimes[index]]
            curatedClusterIDs = curatedClusterIDs + [idx]
    return (curatedSpikeTimeseries, np.array(curatedClusterIDs))

def save_params_as_json(args, save_path, file_name):
    config_filename = os.path.join(save_path, file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(config_filename, 'w') as config_file:
        json.dump(args, config_file, indent=4)
    return(config_filename)

def bound_spike_train(spike_train_list, time_interval, align_to_zero=True, max_neurons=50):
    bounded_spike_trains = []
    for i, spike_train in enumerate(spike_train_list):
        if i > max_neurons:
            return bounded_spike_trains
        bounded = spike_train[np.logical_and(spike_train >= time_interval[0], spike_train < time_interval[1])]
        if align_to_zero:
            bounded = bounded - time_interval[0]
        bounded_spike_trains.append(bounded)
    return bounded_spike_trains

def write_text_file(bound_spikes, filename="spikeData.txt"):
    f = open(filename, "w")
    for i, spike_train in enumerate(bound_spikes):
        for t, spike_time in enumerate(spike_train):
            mssg = "{:.1f}\t{:10.4f}\n".format(i + 1, spike_time)
            f.write(mssg)
    f.close()
    return filename




if __name__ == "__main__":
    main(sys.argv[1:])
