import os
import re
from PIL import Image

def count_frames(frame_folder, pattern=r"frame_\d{4}\.png"):
    # List all files in the directory
    all_files = os.listdir(frame_folder)
    
    # Filter files that match the frame naming pattern
    frame_files = [f for f in all_files if re.match(pattern, f)]
    
    # Sort the frame files to ensure they are in sequence
    frame_files.sort()
    
    # Return the count of frames
    return len(frame_files)



tag = count_frames("/root/autodl-fs/Origin/soccer")

print(tag)