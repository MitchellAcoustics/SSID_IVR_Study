#!/usr/bin/env python3
"""
Hamilton Pipeline with CitySeg Integration

This script runs the Hamilton pipeline with CitySeg integration for the SSID IVR Study.
It processes participant data, converts videos, and performs semantic segmentation
using CitySeg.

Usage:
    uv run python notebooks/run_hamilton_with_cityseg.py
"""

import importlib
import logging
from pathlib import Path
from pprint import pprint

from hamilton import driver
from ivr_utils import ivr_utils, cityseg_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """Run the Hamilton pipeline with CitySeg integration."""
    # Reload modules to ensure latest changes are used
    importlib.reload(ivr_utils)
    importlib.reload(cityseg_utils)

    # Create Hamilton driver
    dr = driver.Builder().with_modules(ivr_utils, cityseg_utils).with_cache().build()

    # Define final variables to compute - including CitySeg processing
    final_vars = [
        "mp4_conversion",
        "output_video_paths",
        "save_gaze_data",
        "cityseg_save_matched_gaze",  # Save matched gaze data
        "cityseg_save_percentages",  # Save percentages data (both NPY and CSV)
        "cityseg_save_analysis_results",  # Terminal node for CitySeg processing
    ]

    # Define inputs - including CitySeg configuration
    inputs = {
        "participant_id": "P43",
        "n_frames_proc": None,
        "eyetracking_dir": "/Volumes/ritd-ag-project-rd01wq-tober63/SSID IVR Study 1/Eyetracking",
        "output_dir": "/Users/mitch/Documents/UCL/Papers_2025/SSID_IVR_Study/data/output/2025-04-15-test",
        "cityseg_config_path": "/Users/mitch/Documents/UCL/Papers_2025/SSID_IVR_Study/cityseg_configs/andrew_config.yaml",
        "frame_step": 1,  # Process every 1st frame
        "batch_size": 5,  # Process 5 frames at a time
        "device": "mps",  # For M2 Mac
    }

    # Visualize the execution graph
    dr.visualize_execution(
        final_vars,
        inputs=inputs,
        output_file_path="graph_with_cityseg.png",
    )

    # Print cache logs
    print("Cache logs:")
    pprint(dr.cache.logs(level="info"))

    # Execute the pipeline
    print("\nExecuting pipeline...")
    try:
        res = dr.execute(final_vars, inputs=inputs)
        print("\nExecution successful!")

        # View the execution results
        print("\nExecution results:")
        run_info = dr.cache.view_run(
            run_id=dr.cache.last_run_id, output_file_path="cityseg_cache_graph.png"
        )
        pprint(run_info)

        return 0
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
