
"""
Video Processing for Eye Tracking Analysis

This script processes eye tracking data from a research study that combines 
video recordings with gaze tracking information for multiple participants. 

Key Functions:
- Converting WMV video files to MP4 format
- Processing eye tracking data from CSV files
- Overlaying gaze positions on video recordings
- Chopping videos based on annotation markers
"""

# %% Import Libraries and Configure Logging
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple
from ivr_utils.ivr_utils import (
    find_participant_files,
    convert_wmv_to_mp4,
    process_video,
)

import logging

# Configure logging with timestamp, level and message
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Participants to process
PARTICIPANTS = ["P48", "P42"]
# Number of frames to process
N_FRAMES_PROC = 30 * 120  # 30 fps * 120 seconds

# Directory
local_data_dir = Path.cwd().parent / "data"
server_data_dir = Path("/Volumes/ritd-ag-project-rd01wq-tober63/SSID IVR Study 1/")
output_dir = local_data_dir.joinpath("output/2025-02-10-test/")

# Check if the server data directory exists
assert server_data_dir.is_dir(), "Server data directory not found"

# Create the output directory if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)

# Eyetracking directory
eyetracking_dir = server_data_dir / "Eyetracking"



# %% Process each participant
for PARTICIPANT_ID in PARTICIPANTS:
    # Find participant files
    part_csv_path, part_wmv_path = find_participant_files(PARTICIPANT_ID, eyetracking_dir)
    print(f"Participant CSV: {part_csv_path}")
    print(f"Participant WMV: {part_wmv_path}")

    # Import csv file for participant
    needed_columns = ["Timestamp", "SlideEvent", "Gaze X", "Gaze Y", "Respondent Annotations active"]
    points = pd.read_csv(part_csv_path, skiprows=lambda x: x < 26, usecols=needed_columns, engine="c")

    # Find the "StartMedia" timestamp
    row = points[points["SlideEvent"] == "StartMedia"]
    timestamp_diff = row["Timestamp"].values[0]

    # Clean the NaN in columns
    points = points.dropna(subset=["Gaze X", "Gaze Y"])

    # Adjust timestamp to start from 0
    points["Timestamp"] = points["Timestamp"] - timestamp_diff

    # Convert WMV to MP4
    fixedfps_result = convert_wmv_to_mp4(
        part_wmv_path,
        output_dir.joinpath(f"output_{PARTICIPANT_ID}_fixedfps.mp4"),
        output_fps=30,
    )
    fixedfps_result.print_status()

    # Paths
    input_video_path = output_dir.joinpath(f"output_{PARTICIPANT_ID}_fixedfps.mp4")
    output_chopping_path = output_dir.joinpath(f"output_chopped_{PARTICIPANT_ID}_fixedfps.mp4")
    output_overlay_path = output_dir.joinpath(f"output_overlay_{PARTICIPANT_ID}_fixedfps.mp4")

    # Process video with chopping and gaze overlay
    process_video(
        input_video_path, 
        output_chopping_path, 
        output_overlay_path, 
        points, 
        N_FRAMES_PROC
    )
