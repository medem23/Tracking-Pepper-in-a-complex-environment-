import os
import glob
import cv2
import pickle
import sys
import random
import torch
import tqdm
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from ByteTrack import Tracker                                     # Use the standard ByteTracker with following par:
                                                                  # track_high_threshold = 0.5,
                                                                  # track_low_threshold = 0.1, match_threshold = 0.9,
                                                                  # new_track_threshold = 0.6, track_buffer = 30
from PIL import Image, ImageDraw, ImageFont


# Paths to be set
path_save =     ''                                                # Path to save the results
path_data_rgb = ''                                                # Path to rgb data folder provided by the challenge
paths_to_sequences = sorted(glob.glob(os.path.join(path_data_rgb, '*')))
path_data_mask = ''                                               # Path to Mask2Former outputs
paths_to_mask2formers = sorted(glob.glob(os.path.join(path_data_mask, '*')))

# Run configuration
name_run = ''                                                     # Name of the run to be set by user
current_training_folder_name = '{}_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), name_run)
run_folder_path = os.path.join(path_save, current_training_folder_name)

# Create main training folder
if not os.path.exists(run_folder_path):
    os.mkdir(run_folder_path)

path_folder_tracking_res = os.path.join(run_folder_path, 'TrackingResults')


if not os.path.exists(path_folder_tracking_res):
    os.mkdir(path_folder_tracking_res)


# Initialise Tracker
track = Tracker()

for path_to_sequence, path_to_mask2former in tqdm.tqdm(zip(paths_to_sequences, paths_to_mask2formers)):
    list_paths_inputs = sorted(glob.glob(os.path.join(path_to_sequence, '*')))
    list_paths_outputs = sorted(glob.glob(os.path.join(path_to_mask2former, '*')))

    predicted_file_path = os.path.join(path_folder_tracking_res, '{}.txt'.format(path_to_sequence[-3:]))

    # Reset tracker
    track.reset()

    for _, data in enumerate(zip(list_paths_inputs, list_paths_outputs)):
        current_path_rgb, current_path_outputs = data

        # Get time frame id
        #frame_id = current_path_rgb.split('/')[-1].split('.')[0]
        
        # For windows:
        frame_id = current_path_rgb.split('\\')[-1].split('.')[0]

        with open(current_path_outputs, 'rb') as f:
            current_output = pickle.load(f)

        bbox = []
        confidence = []

        for key in current_output.keys():
            bbox_values = current_output[key]['bbox']
            bbox_d = current_output[key]['bbox'].tolist()
            diag = round(math.sqrt(bbox_d[2]**2 + bbox_d[3]**2), 1)
            
            # Mask-to-box ratio
            mask_r = current_output[key]['instance_mask']
            mask_r_size = int(np.sum(mask_r))
            area_r = bbox_d[2]*bbox_d[3]
            ratio_r = round(mask_r_size/area_r, 3)

            # Aspect ratio
            box_w = bbox_values[2]
            box_h = bbox_values[3]
            aspect_ratio = round((max(box_w,box_h))/(min(box_w,box_h)), 3)

            # Semantic label
            semlabel = int(current_output[key]['semantic_label'])

            if (ratio_r > 0.1) or (semlabel in {0, 1, 3, 4}) or (( 0.05 < ratio_r <= 0.1) and (aspect_ratio < 2)):
                bbox.append(torch.tensor([bbox_values[0], bbox_values[1], bbox_values[0] + bbox_values[2],
                                          bbox_values[1] + bbox_values[3]]))
                class_conf = 0
                if semlabel in {0, 1, 3, 4}:
                    confidence.append(torch.tensor(0.9999))
                    class_conf = 0.99
                else:
                    confidence.append(torch.tensor(current_output[key]['class_conf']))
                    class_conf = np.round(current_output[key]['class_conf'], 2)

        if bool(bbox):
            box_t = torch.stack(bbox, dim=0)
            conf_t = torch.tensor(confidence)

            input_tracker = [box_t, conf_t]

            for box, score in zip(box_t, conf_t):
                box_n = box.detach().numpy()
        else:
            input_tracker = [torch.zeros(4).unsqueeze(dim=0), torch.zeros(1)]

        tracking_results = track.update(input_tracker)

        # Loop through each detected object
        for _ in range(tracking_results.shape[0]):
            # Extract tracking results for each object in the frame
            x_min, y_min, x_max, y_max = map(lambda v: round(tracking_results[_, v], 2), range(4))
            width, height = round(x_max - x_min, 2), round(y_max - y_min, 2)
            tracker_id = int(tracking_results[_, 5])
            conf_track = round(tracking_results[_, 4], 2)
            
            # Write to file for quantitativ evaluation
            with open(predicted_file_path, 'a') as file:
                file.write("{},{},{},{},{},{},{}\n".format(frame_id, str(tracker_id), str(x_min),
                                                           str(y_min), str(width), str(height), str(conf_track)))


###################################################### Filtering #######################################################

# Define the directories
input_folder = path_folder_tracking_res
output_folder = os.path.join(run_folder_path, 'FilteredResults')

#define filtering threshold
threshold = 0.975
min_len = 5

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the files 400.txt to 475.txt
# Please adjust this for other datasets (here: evaluation dataset)
for i in range(400, 476):
    input_file = os.path.join(input_folder, f'{i}.txt')
    
    # Load the tracking data
    data = pd.read_csv(input_file, header=None, names=['frameid', 'trackingid', 'bbox_x_topleft', 'bbox_y_topleft',
                                                       'width', 'height', 'confidence'])

    # Group by trackingid
    grouped = data.groupby('trackingid')

    #define filter function
    def filter_tracks(group):
        if group['confidence'].max() < threshold:   # confidence filtering
            return False
        elif len(group) < min_len:                  # length filtering
            return False
        return True

    # Apply the filter function
    filtered_data = grouped.filter(filter_tracks)

    output_file = os.path.join(output_folder, f'{i}.txt')
    
    # Save the filtered data
    filtered_data.to_csv(output_file, index=False, header=False)

    print(f"Filtering complete for {i}.txt")

print("All files processed.")
