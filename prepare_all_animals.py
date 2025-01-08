import os
from utils import recording_id_list

def main():
    print(recording_id_list)
    command = "python prepare_data_new.py --max-fano-factor 12.0 --min-fano-factor 0.5 --time_span Short_awake"
    for recording_id in recording_id_list:
        spec_command = command + " --mouse_implant_recording " + recording_id
        spec_command += " --output_filename "+recording_id
        spec_command += " --save_path ./data/preparedData/all_animals/"
        spec_command += " --align_to_zero True"
        os.system(spec_command)
    
    
if __name__ == "__main__":
    main()