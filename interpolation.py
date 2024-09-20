import os
import numpy as np
########################################Relabeling##########################################################################
"""
In order to perform interpolation, we need to relabel the frame ids,
such that the frame ids evolve in an ascending order with step size 1 (1,2,3,...)
"""
# rgb_dir: rgb directory containing all video sequences
rgb_dir = r''

# Path to the filtered track directory
filtered_dir = r''

# frame_dict: dictionary of dictionaries in order to map the old frame id to the relabeled frame id for each video sequence
frame_dict = {}

# Video Sequence numbers, please adjust for your dataset (here: Evaluation Dataset)
first_seq = 400
last_seq = 475

# Iterate over each sequence 
for seq_num in range(first_seq, last_seq+1): #Please adjust this for other datasets
    #seq_dir: directory of individual video sequence
    seq_dir = os.path.join(rgb_dir, str(seq_num))
    
    if os.path.isdir(seq_dir):  # Check if the directory exists
        # List all the .tiff files (frames) in the directory
        frames = [f for f in os.listdir(seq_dir) if f.endswith('.tiff')]
        
        # Sort frames by their filename
        frames.sort()
        
        # Initialize the dictionary for the current video sequence
        frame_dict[seq_num] = {}
        
        # Create mapping: original frame id <-> relabeled frame id
        for idx, frame in enumerate(frames, start=1):
            frame_id = os.path.splitext(frame)[0]  # Remove the .tiff extension
            frame_dict[seq_num][frame_id] = idx


"""
Having created the dictionary, the relabeling can be performed:
"""

# new directory with relabeled tracking results
relabeled_dir = os.path.join(os.path.dirname(filtered_dir), 'relabeled')

# Create the relabeled directory if it doesn't exist
os.makedirs(relabeled_dir, exist_ok=True)

# Iterate over each sequence
for seq_num in range(first_seq, last_seq+1):
    
    seq_file = os.path.join(filtered_dir, f'{seq_num}.txt')
    
    relabeled_file = os.path.join(relabeled_dir, f'{seq_num}.txt')
    
    # Open the original file and the new file for writing
    with open(seq_file, 'r') as infile, open(relabeled_file, 'w') as outfile:
        for line in infile:
            # Split the line into its components
            parts = line.strip().split(',')
            original_frameid = parts[0]
            
            # Get the new (relabeled) frameid from the dictionary
            new_frameid = frame_dict[seq_num][original_frameid]
            
            # Replace the old frameid with the new one
            if new_frameid is not None:
                parts[0] = str(new_frameid)
            
            # Write the modified line to the new file
            outfile.write(','.join(parts) + '\n')

print("Relabeling complete! The relabeled files are stored in the 'relabeled' directory.")


########################################Interpolation########################################################################

# Function to perform linear interpolation between two bounding boxes
def interpolate_bbox(start_frame, end_frame, start_bbox, end_bbox):
    interpolated = []
    total_frames = end_frame - start_frame
    for i in range(1, total_frames):
        factor = i / total_frames
        interpolated_bbox = [
            round(start_bbox[0] + factor * (end_bbox[0] - start_bbox[0]), 1),
            round(start_bbox[1] + factor * (end_bbox[1] - start_bbox[1]), 1),
            round(start_bbox[2] + factor * (end_bbox[2] - start_bbox[2]), 1),
            round(start_bbox[3] + factor * (end_bbox[3] - start_bbox[3]), 1)
        ]
        interpolated.append(interpolated_bbox)
    return interpolated

# New directory with interpolated tracking results
interpolated_dir = os.path.join(os.path.dirname(filtered_dir), 'interpolated')

# Create the interpolated directory if it doesn't exist
os.makedirs(interpolated_dir, exist_ok=True)

# Iterate over each sequence
for seq_num in range(first_seq, last_seq+1):
    
    seq_file = os.path.join(relabeled_dir, f'{seq_num}.txt')
    
    # Read the contents of the file
    with open(seq_file, 'r') as infile:
        data = [line.strip().split(',') for line in infile]
    
    # tracks: dictionary with trackid as key and list as value (list contains frameids, bboxes and confidence values for corresponding track)
    tracks = {}
    for line in data:
        frameid = int(line[0])
        trackid = int(line[1])
        bbox = list(map(float, line[2:6]))  # bbox_x, bbox_y, bbox_width, bbox_height
        
        if trackid not in tracks:
            tracks[trackid] = []
        tracks[trackid].append((frameid, bbox, line[6]))  # Append (frameid, bbox, confidence)
    
    # interpolated tracks
    interpolated_tracks = []

    #iterate over tracks
    for trackid, track_data in tracks.items():
        # Sort the track data by frameid
        track_data.sort(key=lambda x: x[0])
        
        #iterate over frames of the track
        for i in range(len(track_data) - 1):
            current_frame, current_bbox, current_conf = track_data[i]
            next_frame, next_bbox, next_conf = track_data[i + 1]
            
            # Add current detection
            interpolated_tracks.append([current_frame, trackid, *current_bbox, current_conf])
            
            # Check if there is a gap between the current and next frame
            # next_frame - current_frame > 1: checks whether there is a gap
            # next_frame - current_frame < gap_size_limit: limits the gap_size to avoid interpolation of False-Positive detections
            gap_size_limit = 4  # This is just a conservatively chosen value, please adjust this for future applications
            if (next_frame - current_frame) > 1 and (next_frame - current_frame) < gap_size_limit:
                
                # Interpolate the bounding boxes for the missing frames
                interpolated_bboxes = interpolate_bbox(current_frame, next_frame, current_bbox, next_bbox)
                for j, bbox in enumerate(interpolated_bboxes, start=1):
                    interpolated_tracks.append([current_frame + j, trackid, *bbox, current_conf])
        
        # Add the last frame of the track
        interpolated_tracks.append([next_frame, trackid, *next_bbox, next_conf])
    
    # Sort interpolated tracks by frameid to ensure the correct order
    interpolated_tracks.sort(key=lambda x: x[0])

    # Write the interpolated data to the new file
    interpolated_file = os.path.join(interpolated_dir, f'{seq_num}.txt')
    with open(interpolated_file, 'w') as outfile:
        for line in interpolated_tracks:
            outfile.write(','.join(map(str, line)) + '\n')

print("Interpolation complete! The interpolated files are stored in the 'interpolated' directory.")



########################################Inverse Relabeling####################################################################
"""
Finally, the original frameids have to be restored in the file, so we have to reverse the relabeling.
"""

# Directory to store the final text files with original frame IDs
final_dir = os.path.join(os.path.dirname(filtered_dir), 'final')

os.makedirs(final_dir, exist_ok=True)

# Create an inverse mapping from the frame_dict
inverse_frame_dict = {}

for seq_num, mapping in frame_dict.items():
    inverse_frame_dict[seq_num] = {v: k for k, v in mapping.items()}

# Iterate over each sequence
for seq_num in range(first_seq, last_seq+1):
    # Path to the current interpolated sequence file
    interpolated_file = os.path.join(interpolated_dir, f'{seq_num}.txt')
    
    # Path to the new final file with original frame IDs
    final_file = os.path.join(final_dir, f'{seq_num}.txt')
    
    # Open the interpolated file and the new final file for writing
    with open(interpolated_file, 'r') as infile, open(final_file, 'w') as outfile:
        for line in infile:
            # Split the line into its components
            parts = line.strip().split(',')
            ordered_frameid = int(parts[0])
            
            # Get the original frameid from the inverse dictionary
            original_frameid = inverse_frame_dict[seq_num][ordered_frameid]
            
            if original_frameid is not None:
                # Replace the ordered frameid with the original one
                parts[0] = original_frameid
            
            # Write the modified line to the new final file
            outfile.write(','.join(parts) + '\n')

print("Inverse Relabeling complete! The final interpolated files with original frame IDs are stored in the 'final' directory.")