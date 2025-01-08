import os 
from datetime import datetime 

utils_path = os.path.realpath(__file__)
figure_directory = os.path.join(os.path.abspath(os.path.join(utils_path,"..")),"figures/")
print("Figure directory set to: ", figure_directory)
def save_figure(fig,name="",save_types=['svg','png']):
    """saves a figure by date (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        name (str, optional): name to be saved as. Current time will be appended to this

        REMEMBER in three months time you wont remember the day you made a plot...so if you think its important rememeber to move it. This is a kind of backup / last resort 
    """	
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)

    #make today-specific directory inside figure directory  
    today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(figure_directory + f"{today}/"):
        os.mkdir(figure_directory + f"{today}/")
    
    for filetype in save_types:
        figdir = figure_directory + f"{today}/"
        now = datetime.strftime(datetime.now(),'%H%M')
        path_ = f"{figdir}{name}_{now}"
        path = path_
        i=1
        while True:
            if os.path.isfile(path+"." + filetype):
                path = path_+"_"+str(i)
                i+=1
            elif i >= 100: break
            else: break
        fig.savefig(path+"."+filetype,bbox_inches='tight')
    return path

recording_id_list = [
                    "136_1_2",
                    "136_1_3",
                    "136_1_4",
                    "148_2_2",
                    "149_1_1",
                    "149_1_2",
                    "149_1_3",
                    "149_1_4",
                    "149_2_1",
                    "162_1_3",
                    "178_1_1",
                    "178_1_2",
                    "178_1_3",
                    "178_1_4",
                    "178_1_5",
                    "178_1_6",
                    "178_1_7",
                    "178_1_8",
                    "178_1_9",
                    "178_2_1",
                    "178_2_2",
                    "178_2_3",
                    "178_2_4",
                     ]

def recording_to_directory(recording_id):
    """Take a recording string of form... 
            "ratID_implantID_recording_ID" eg "178_1_3" 
    ...and returns a full directory corresponding for this recording (i.e. the directory containing ephys, video, behav_sync, post_process_ppseq etc.)

    Should work if local OR cluster

    Args:
        recording (_type_): _description_
    """    

    if os.path.isdir("/nfs/winstor/sjones/"):
        head = "/nfs/winstor/sjones/"
    elif os.path.isdir("/Volumes/sjones/"):
        head = "/Volumes/sjones/"
    else:
        print("Cannot locate `sjones` directory at either `/Volumes/sjones/` or `/nfs/winstor/sjones/`")
    
    mouse_implant_recording = recording_id.split("_")
    mouse = mouse_implant_recording[0]
    implant = mouse_implant_recording[1]
    recording = mouse_implant_recording[2]

    animal_directory = os.path.join(head,"projects/sequence_squad/organised_data/animals/")
    implant_directory = os.path.join(animal_directory,"EJT" + mouse + "_implant" + implant)
    assert(os.path.isdir(implant_directory)), f"{implant_directory} is not a valid directory"
    
    recording_found = False
    for rel_recording_directory in os.listdir(implant_directory):
        if "recording" + recording in rel_recording_directory:
            recording_directory = os.path.join(implant_directory,rel_recording_directory)
            recording_found = True
            continue
    assert recording_found == True, f"No directory containing `recording{recording}` found within {implant_directory}"

    return recording_directory


def select_within_timespan(times,time_span,align_to_zero=False):
    """times is a list of arrays (e.g. spike times for many neurons).
    This func filters out those which are not within any of the spans listed in time_span (a list_of_lists)
    
    times: list of arrays e.g. [np.array([1,4,6]), np.array([1,5,9]), ...]
    time_spans: list of lists e.g. [[100,150],[2000,3010]]"""
    time_slices = []
    concatenation_shift = 0
    for time_span_ in time_span:
        time_slice = [el[(el>time_span_[0]) & (el<time_span_[1])] for el in times]
        if align_to_zero is True:
            shiftby = time_span_[0]-concatenation_shift
            time_slice = [el - shiftby for el in time_slice]
            concatenation_shift += time_span_[1] - time_span_[0]
        time_slices = time_slices + [time_slice]
    times = []
    for i in range(len(time_slices[0])):
        time_list = []
        for j in range(len(time_slices)):
            time_list = time_list + list(time_slices[j][i])
        times += [time_list]
    total_time = sum([span[1] - span[0] for span in time_span])
    return times, total_time