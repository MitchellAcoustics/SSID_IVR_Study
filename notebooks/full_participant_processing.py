# %%

import pandas as pd
from pathlib import Path
from ivr_utils.ivr_utils import (
    participant_file_paths,
    mp4_conversion,
    process_video,
)
import numpy as np
import logging
import cityseg as cs


def process_participant(
    participant_id: str,
    eyetracking_dir: Path,
    output_dir: Path,
    n_frames_proc: int = None,
    cityseg_config: str = None,
    # Optional parameters for CitySeg
    cityseg_params: dict = None,
):
    """
    Process the participant's data.

    Args:
        participant_id (str): The participant ID.
        eyetracking_dir (Path): The directory containing eyetracking data.
        output_dir (Path): The directory to save the processed data.
        n_frames_proc (int, optional): Number of frames to process. Defaults to None.
    """
    # Check directories
    eyetracking_dir, output_dir = _check_dirs(eyetracking_dir, output_dir)

    # Find participant files
    part_csv_path, part_wmv_path = participant_file_paths(
        participant_id, eyetracking_dir
    )

    logging.info(f"Participant CSV: {part_csv_path}")
    logging.info(f"Participant WMV: {part_wmv_path}")
    points = _participant_csv(part_csv_path)

    # Convert WMV to MP4
    mp4_path = output_dir.joinpath(f"{participant_id}_fixedfps.mp4")
    if mp4_path.exists():
        logging.info(f"MP4 file already exists: {mp4_path}")
    else:
        logging.info(f"Converting WMV to MP4: {mp4_path}")
        # Convert WMV to MP4 with fixed FPS
        mp4_result = mp4_conversion(part_wmv_path, mp4_path, output_fps=30)
        logging.info(f"Converted MP4: {mp4_path}")

    # Chopping and Overlaying Gaze point
    output_chopping_path = output_dir.joinpath(f"{participant_id}_chopped.mp4")
    output_overlay_path = output_dir.joinpath(f"{participant_id}_overlay.mp4")
    if output_chopping_path.exists() and output_overlay_path.exists():
        logging.info(
            f"Chopped and overlayed video files already exist: {output_chopping_path}, {output_overlay_path}"
        )
    else:
        logging.info(
            f"Chopping and overlaying video: {mp4_path} -> {output_chopping_path}, {output_overlay_path}"
        )
        # Process video with chopping and gaze overlay
        process_video(
            mp4_path, output_chopping_path, output_overlay_path, points, n_frames_proc
        )
        logging.info(
            f"Processed video: {mp4_path} -> {output_chopping_path}, {output_overlay_path}"
        )

    logging.info(f"Chopped video: {output_chopping_path}")
    logging.info(f"Overlay video: {output_overlay_path}")

    # Create CitySeg configuration
    config = None
    if cityseg_config:
        config = cs.Config.from_yaml(cityseg_config)
        config.input = output_chopping_path
        config.input_type = config._determine_input_type()
        config.output_dir = output_dir.joinpath(f"{output_chopping_path.stem}_cityseg")

    if config is None:
        config = _create_cityseg_config(
            output_chopping_path,
            output_dir,
            cityseg_params.get("model_name", "default_model"),
        )

    run_cityseg(config)
    logging.info(f"CitySeg config: {config}")


def _check_dirs(
    eyetracking_dir: Path,
    output_dir: Path,
):
    """
    Check if the directories exist and create the output directory if it does not exist.

    Args:
        eyetracking_dir (Path): Eyetracking directory.
        output_dir (Path): Output directory.
    """
    # Check if the eyetracking directory exists
    assert eyetracking_dir.is_dir(), "Eyetracking directory not found"

    # Create the output directory if it does not exist
    if not output_dir.is_dir():
        logging.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"Output directory already exists: {output_dir}")

    return eyetracking_dir, output_dir


def _participant_csv(part_csv_path: Path):
    points = pd.read_csv(part_csv_path, usecols=[0], engine="c")
    row_index = points[points.iloc[:, 0] == "#DATA"].index[0]

    points = pd.read_csv(
        part_csv_path, skiprows=lambda x: x < row_index + 2, engine="c"
    )

    # find the "StartMedia" timestamp
    row = points[points["SlideEvent"] == "StartMedia"]
    timestamp_diff = row["Timestamp"].values[0]

    clean = ["ET_GazeLeftx", "ET_GazeRightx", "ET_GazeLefty", "ET_GazeRighty"]
    points[clean] = points[clean].replace(-1, np.nan)

    # check if the file does not have Gaze X and Gaze Y columns, if not calculate it with ET_Gaze columns
    if "Gaze X" not in points.columns:
        points["Gaze X"] = points[["ET_GazeLeftx", "ET_GazeRightx"]].mean(axis=1)

    if "Gaze Y" not in points.columns:
        points["Gaze Y"] = points[["ET_GazeLefty", "ET_GazeRighty"]].mean(axis=1)

    # Clean the NaN in columns
    points = points.dropna(subset=["Gaze X", "Gaze Y"])

    # Adjust timestamp to start from 0
    points["Timestamp"] = points["Timestamp"] - timestamp_diff

    return points


def _create_cityseg_config(chopped_video_path: Path, output_dir: Path, model_name: str):
    """
    Create a configuration object for CitySeg.
    Args:
        chopped_video_path (Path): Path to the chopped video.
        output_dir (Path): Output directory.
        model (str): Model to be used for CitySeg.
    Returns:
        cs.Config: Configuration object.
    """
    model_config = cs.config.ModelConfig(
        name=model_name,
    )

    # Create a configuration object
    config = cs.Config()

    # Set the parameters for the configuration
    config.set_parameters(
        input=chopped_video_path,
        output_dir=output_dir.joinpath(f"{chopped_video_path.stem}_cityseg."),
        model=model_config,
    )

    return config


def run_cityseg(config: cs.Config):
    # Create processor
    processor = cs.create_processor(config)

    # Process the data
    processor.process()
    logging.info("CitySeg processing completed.")


def main():
    """Command-line interface for processing participant data."""
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Process participant data from eyetracking recordings."
    # )
    # parser.add_argument("participant_id", type=str, help="The participant ID")
    # parser.add_argument(
    #     "eyetracking_dir", type=str, help="The directory containing eyetracking data"
    # )
    # parser.add_argument(
    #     "output_dir", type=str, help="The directory to save the processed data"
    # )
    # parser.add_argument(
    #     "--n_frames_proc",
    #     type=int,
    #     default=None,
    #     help="Number of frames to process (default: all frames)",
    # )
    # parser.add_argument(
    #     "--cityseg_config",
    #     type=str,
    #     default=None,
    #     help="Path to CitySeg configuration YAML file",
    # )
    # parser.add_argument(
    #     "--cityseg_params",
    #     type=str,
    #     default=None,
    #     help="JSON string of additional CitySeg parameters",
    # )

    # args = parser.parse_args()

    # # Convert string paths to Path objects
    # eyetracking_dir = Path(args.eyetracking_dir)
    # output_dir = Path(args.output_dir)

    # # Parse CitySeg parameters if provided
    # cityseg_params = None
    # if args.cityseg_params:
    #     import json

    #     try:
    #         cityseg_params = json.loads(args.cityseg_params)
    #     except json.JSONDecodeError:
    #         logging.error("Invalid JSON format for cityseg_params")
    #         return 1

    # Configure logging with timestamp, level and message
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the participant ID and directories
    participant_id = "P43"
    n_frames_proc = None
    eyetracking_dir = Path(
        "/Volumes/ritd-ag-project-rd01wq-tober63/SSID IVR Study 1/Eyetracking"
    )
    output_dir = Path.cwd() / "data/output/2025-04-01-test/"
    cityseg_config = Path.cwd() / "cityseg_configs/andrew_config.yaml"

    # Process the participant data
    process_participant(
        participant_id=participant_id,
        eyetracking_dir=eyetracking_dir,
        output_dir=output_dir,
        n_frames_proc=n_frames_proc,
        cityseg_config=cityseg_config,
        cityseg_params=None,
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
