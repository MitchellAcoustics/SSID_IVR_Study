"""IVR Study Utilities.

This module provides utilities for processing and analyzing data from IVR (Immersive Virtual Reality)
studies, specifically handling eyetracking data and video processing tasks.

Key Features:
    - File management for participant data
    - Video conversion (WMV to MP4) with hardware acceleration
    - Eyetracking data processing
    - Video validation and integrity checks

Requirements:
    - OpenCV (cv2)
    - av (PyAV)
    - ffmpeg with videotoolbox support
    - Python 3.10+
    - macOS with Apple Silicon (for hardware acceleration)
"""

from collections.abc import Generator, Callable
import contextlib
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TypeVar
import cv2  # For video processing and validation
import math  # For frame count comparison
import subprocess  # For ffmpeg operations
import logging  # For operation logging
import av  # For advanced video operations
import pandas as pd
from tqdm.auto import trange
import numpy as np
import hashlib  # For file checksums
import time  # For performance logging

from hamilton.function_modifiers import datasaver, cache, tag
from hamilton.io import utils

logger = logging.getLogger(__name__)

# --- Configuration ---

# FFMPEG encoding configuration
FFMPEG_CONFIG = {
    "video_codec": "h264_videotoolbox",
    "target_bitrate": "5M",
    "max_bitrate": "7M",
    "buffer_size": "10M",
    "profile": "high",
    "color_range": "1",
}

# --- Type Aliases ---

# Path-like objects (str or Path)
PathLike = str | Path

# Frame processor callback type
FrameProcessor = Callable[[np.ndarray, int, dict], np.ndarray | None]

# File path tuple
FilePathTuple = tuple[str, str]

# Type variable for generic functions
T = TypeVar("T")

# --- Path Utilities ---


def _ensure_path(path: PathLike) -> Path:
    """Convert string path to Path object if needed.

    Args:
        path: Path as string or Path object

    Returns:
        Path object
    """
    return Path(path) if isinstance(path, str) else path


def participant_id_sanitized(participant_id: str) -> str:
    """Sanitize participant ID for use in filenames.

    Args:
        participant_id: Raw participant ID

    Returns:
        Sanitized participant ID
    """
    return participant_id.strip().upper()


# --- File Discovery ---


@tag(category="file_discovery")
def participant_file_paths(
    participant_id: str, eyetracking_dir: PathLike
) -> FilePathTuple:
    """
    Find the participant files for the given participant ID in the given data directory.

    This function searches for two specific files associated with a participant:
    1. A CSV file containing eyetracking data.
    2. A WMV video file located in the "Screen Recording" subdirectory.

    The function ensures that exactly one CSV file and one WMV video file are found.
    If either file is not found or if multiple files are found, an assertion error is raised.

    Args:
        participant_id: The ID of the participant whose files are to be found.
        eyetracking_dir: The directory where the participant files are located.

    Returns:
        A tuple containing the resolved paths to the participant's CSV file and WMV video file.

    Raises:
        AssertionError: If the CSV file or the WMV video file is not found or if multiple files are found.
    """
    start_time = time.time()
    data_dir = _ensure_path(eyetracking_dir)
    sanitized_id = participant_id_sanitized(participant_id)

    logger.info(f"Searching for files for participant {sanitized_id} in {data_dir}")

    # Find eyetracking csv
    part_csv_l = list(data_dir.rglob(f"*{sanitized_id}.csv"))
    assert len(part_csv_l) == 1, (
        f"Participant CSV not found or more than one found for {sanitized_id}"
    )

    part_csv_path = part_csv_l[0].resolve()
    logger.debug(f"Found CSV file: {part_csv_path}")

    # Find eyetracking video
    scenario_vid_dir = part_csv_path.parents[1] / "Screen Recording"
    part_vid_l = list(scenario_vid_dir.rglob(f"*{sanitized_id}_*.wmv"))
    assert len(part_vid_l) == 1, (
        f"Participant video not found or more than one found for {sanitized_id}"
    )

    part_vid_path = part_vid_l[0].resolve()
    logger.debug(f"Found WMV file: {part_vid_path}")

    elapsed = time.time() - start_time
    logger.debug(f"File discovery completed in {elapsed:.2f}s")

    return part_csv_path.as_posix(), part_vid_path.as_posix()


# --- Eyetracking Data Processing ---


def eyetracking_csv_path(participant_file_paths: FilePathTuple) -> str:
    """Extract CSV path from participant file paths.

    Args:
        participant_file_paths: Tuple of (csv_path, wmv_path)

    Returns:
        Path to CSV file
    """
    csv_path, _ = participant_file_paths
    return csv_path


def wmv_video_path(participant_file_paths: FilePathTuple) -> str:
    """Extract WMV path from participant file paths.

    Args:
        participant_file_paths: Tuple of (csv_path, wmv_path)

    Returns:
        Path to WMV file
    """
    _, wmv_path = participant_file_paths
    return wmv_path


def raw_eyetracking_data(eyetracking_csv_path: str) -> pd.DataFrame:
    """Load raw eyetracking data from CSV file.

    Args:
        eyetracking_csv_path: Path to eyetracking CSV file

    Returns:
        DataFrame with raw eyetracking data
    """
    csv_path = _ensure_path(eyetracking_csv_path)
    logger.info(f"Loading raw eyetracking data from {csv_path}")

    # First read to find the data section
    points = pd.read_csv(csv_path, usecols=[0], engine="c")
    row_index = points[points.iloc[:, 0] == "#DATA"].index[0]

    # Second read to load actual data
    points = pd.read_csv(csv_path, skiprows=lambda x: x < row_index + 2, engine="c")

    logger.debug(f"Loaded {len(points)} rows of raw eyetracking data")
    return points


def start_media_timestamp(raw_eyetracking_data: pd.DataFrame) -> float:
    """Find the StartMedia timestamp in eyetracking data.

    Args:
        raw_eyetracking_data: Raw eyetracking data

    Returns:
        StartMedia timestamp
    """
    row = raw_eyetracking_data[raw_eyetracking_data["SlideEvent"] == "StartMedia"]
    assert len(row) > 0, "StartMedia event not found in eyetracking data"
    timestamp = row["Timestamp"].values[0]
    logger.debug(f"StartMedia timestamp: {timestamp}")
    return timestamp


def cleaned_gaze_columns(raw_eyetracking_data: pd.DataFrame) -> pd.DataFrame:
    """Clean gaze columns by replacing -1 with NaN using vectorized operations.

    Args:
        raw_eyetracking_data: Raw eyetracking data

    Returns:
        DataFrame with cleaned gaze columns
    """
    df = raw_eyetracking_data.copy()

    # Define columns to clean in one operation
    clean_cols = ["ET_GazeLeftx", "ET_GazeRightx", "ET_GazeLefty", "ET_GazeRighty"]

    # Vectorized replacement of -1 with NaN
    df[clean_cols] = df[clean_cols].replace(-1, np.nan)

    return df


def computed_gaze_coordinates(cleaned_gaze_columns: pd.DataFrame) -> pd.DataFrame:
    """Compute Gaze X and Y coordinates if not present using vectorized operations.

    Args:
        cleaned_gaze_columns: DataFrame with cleaned gaze columns

    Returns:
        DataFrame with Gaze X and Y columns
    """
    df = cleaned_gaze_columns.copy()

    # Calculate X and Y gaze coordinates in single vectorized operations
    if "Gaze X" not in df.columns:
        df["Gaze X"] = df[["ET_GazeLeftx", "ET_GazeRightx"]].mean(axis=1)
        logger.debug("Computed Gaze X from left/right eye data")

    if "Gaze Y" not in df.columns:
        df["Gaze Y"] = df[["ET_GazeLefty", "ET_GazeRighty"]].mean(axis=1)
        logger.debug("Computed Gaze Y from left/right eye data")

    return df


def normalized_timestamps(
    computed_gaze_coordinates: pd.DataFrame, start_media_timestamp: float
) -> pd.DataFrame:
    """Normalize timestamps to start from 0 at StartMedia event.

    Args:
        computed_gaze_coordinates: DataFrame with gaze coordinates
        start_media_timestamp: StartMedia timestamp

    Returns:
        DataFrame with normalized timestamps
    """
    df = computed_gaze_coordinates.copy()
    df["Timestamp"] = df["Timestamp"] - start_media_timestamp
    logger.debug(f"Normalized timestamps by subtracting {start_media_timestamp}")
    return df


def filtered_eyetracking_data(normalized_timestamps: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows with NaN in Gaze X or Y.

    Args:
        normalized_timestamps: DataFrame with normalized timestamps

    Returns:
        Filtered DataFrame
    """
    df = normalized_timestamps.dropna(subset=["Gaze X", "Gaze Y"])
    logger.debug(
        f"Filtered out {len(normalized_timestamps) - len(df)} rows with NaN gaze coordinates"
    )
    return df


def participant_points(participant_file_paths: FilePathTuple) -> pd.DataFrame:
    """
    Process participant eyetracking data from CSV file.

    This function is a composite of the individual processing steps and is maintained
    for backward compatibility.

    Args:
        participant_file_paths: Tuple of (csv_path, wmv_path)

    Returns:
        Processed eyetracking data
    """
    csv_path = eyetracking_csv_path(participant_file_paths)
    raw_data = raw_eyetracking_data(csv_path)
    start_timestamp = start_media_timestamp(raw_data)
    cleaned_data = cleaned_gaze_columns(raw_data)
    with_gaze = computed_gaze_coordinates(cleaned_data)
    normalized_data = normalized_timestamps(with_gaze, start_timestamp)
    filtered_data = filtered_eyetracking_data(normalized_data)

    return filtered_data


# --- Video Processing ---


class ConversionStatus(Enum):
    """Video conversion status indicators.

    Attributes:
        SUCCESS: Conversion completed successfully
        ALREADY_EXISTS: Valid converted file already exists
        FAILED: Conversion failed with error
    """

    SUCCESS = auto()
    ALREADY_EXISTS = auto()
    FAILED = auto()


@dataclass
class ConversionResult:
    """Result of video conversion operation.

    Attributes:
        status: Conversion status
        output_str: Path to output file as string
        error_message: Error message if conversion failed
    """

    status: ConversionStatus
    output_str: str
    error_message: str | None = None

    @property
    def output_path(self) -> Path:
        """Get output path as Path object."""
        return _ensure_path(self.output_str)

    def print_status(self) -> None:
        """Print human-readable status message."""
        if self.status == ConversionStatus.SUCCESS:
            print("Conversion successful")
        elif self.status == ConversionStatus.ALREADY_EXISTS:
            print("MP4 file already exists and is valid")
        elif self.status == ConversionStatus.FAILED:
            print(f"Conversion failed: {self.error_message}")


@dataclass
class VideoInfo:
    """Information about a video file.

    Attributes:
        frame_count: Number of frames
        fps: Frames per second
        duration: Duration in seconds
    """

    frame_count: int
    fps: float
    duration: float


@contextlib.contextmanager
def video_capture(path: PathLike) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager for video capture operations.

    Args:
        path: Path to video file

    Yields:
        OpenCV VideoCapture object

    Raises:
        ValueError: If video cannot be opened
    """
    path_obj = _ensure_path(path)
    cap = cv2.VideoCapture(path_obj.as_posix())

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path_obj}")

    try:
        yield cap
    finally:
        cap.release()


def _get_video_info(video_path: Path) -> VideoInfo | None:
    """Extract frame count, FPS and duration from video file.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object or None if video cannot be opened
    """
    start_time = time.time()
    logger.debug(f"Getting video info for {video_path}")

    try:
        with video_capture(video_path) as cap:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            info = VideoInfo(frame_count, fps, duration)
            logger.debug(
                f"Video info: {frame_count} frames, {fps} fps, {duration:.2f}s"
            )
    except ValueError as e:
        logger.warning(f"Failed to open video file: {video_path} - {e}")
        return None

    elapsed = time.time() - start_time
    logger.debug(f"Got video info in {elapsed:.2f}s")

    return info


def _validate_video(
    video_path: Path, source_path: Path | None = None, full_check: bool = False
) -> tuple[bool, str | None]:
    """
    Validates video file integrity and optionally compares it with a source video.

    Args:
        video_path: The path to the video file to be validated.
        source_path: The path to the source video file for comparison (optional).
        full_check: If True, performs a full check including comparison with the source video.

    Returns:
        A tuple containing a boolean indicating if the video is valid,
        and an optional error message if the video is not valid.
    """
    start_time = time.time()
    logger.debug(f"Validating video: {video_path}")

    info = _get_video_info(video_path)
    if not info:
        return False, "Failed to open video file"

    if info.frame_count <= 0:
        return False, "Invalid frame count"

    if full_check and source_path:
        source_info = _get_video_info(source_path)
        if not source_info:
            return False, "Failed to read source video"

        if not math.isclose(source_info.duration, info.duration, rel_tol=0.1):
            return (
                False,
                f"Duration mismatch: source={source_info.duration:.2f}s, output={info.duration:.2f}s",
            )

    elapsed = time.time() - start_time
    logger.debug(f"Video validation completed in {elapsed:.2f}s")

    return True, None


def with_file_operation(path: PathLike, operation: Callable[[Path], T]) -> T:
    """Execute an operation with a file path, ensuring it exists.

    Args:
        path: Path to file
        operation: Function to execute with the path

    Returns:
        Result of operation

    Raises:
        FileNotFoundError: If file does not exist
    """
    path_obj = _ensure_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return operation(path_obj)


def _calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA-256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA-256 checksum as hex string
    """

    def _checksum(path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    return with_file_operation(file_path, _checksum)


def mp4_output_path(participant_id: str, output_dir: PathLike) -> str:
    """Generate output path for MP4 file.

    Args:
        participant_id: Participant ID
        output_dir: Output directory

    Returns:
        Path to output MP4 file
    """
    sanitized_id = participant_id_sanitized(participant_id)
    output_path = _ensure_path(output_dir).joinpath(f"{sanitized_id}_fixedfps.mp4")
    return output_path.as_posix()


def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    config: dict[str, str] = FFMPEG_CONFIG,
    output_fps: int | None = None,
) -> list[str]:
    """Build ffmpeg command with configuration.

    Args:
        input_path: Input video path
        output_path: Output video path
        config: FFMPEG configuration dictionary
        output_fps: Optional output FPS

    Returns:
        List of command arguments
    """
    cmd = [
        "ffmpeg",
        # Hardware acceleration for Apple Silicon
        "-hwaccel",
        "videotoolbox",
        "-i",
        input_path,
        # Video encoding settings from config
        "-c:v",
        config["video_codec"],
        "-b:v",
        config["target_bitrate"],
        "-maxrate",
        config["max_bitrate"],
        "-bufsize",
        config["buffer_size"],
        "-profile:v",
        config["profile"],
        "-movflags",
        "+faststart",  # Enable streaming optimization
        "-color_range",
        config["color_range"],  # Full color range
    ]

    # FPS settings
    if output_fps is not None:
        cmd.extend(["-r", str(output_fps)])

    # No audio, overwrite, etc.
    cmd.extend(
        [
            "-an",  # no audio
            "-y",  # overwrite output file
            "-stats",  # show progress
            "-v",
            "warning",  # show warnings
            output_path,
        ]
    )

    return cmd


def check_output_file_exists(file_path: PathLike) -> bool:
    """Check if an output file exists and is valid.

    This is a utility function for checking if we need to re-run processing.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and is valid
    """
    path = _ensure_path(file_path)
    if not path.exists():
        return False

    # For video files, validate them
    if path.suffix.lower() in [".mp4", ".wmv"]:
        is_valid, _ = _validate_video(path)
        return is_valid

    # For numpy files, try to load them
    if path.suffix.lower() == ".npy":
        try:
            np.load(path)
            return True
        except Exception:
            return False

    # Default case, just check if file exists and has size > 0
    return path.exists() and path.stat().st_size > 0


@tag(version="1.0", category="video_conversion")
@cache()
def mp4_conversion(
    participant_file_paths: FilePathTuple,
    participant_id: str,
    output_dir: PathLike,
    output_fps: int | None = None,
    full_validation: bool = True,
) -> ConversionResult:
    """Convert WMV video to MP4 format using Apple Silicon hardware acceleration.

    This function converts WMV videos to MP4 format using the h264_videotoolbox codec
    for hardware acceleration on Apple Silicon. It includes validation of the output
    file and handles existing files intelligently.

    Args:
        participant_file_paths: Tuple of (csv_path, wmv_path)
        participant_id: Participant ID
        output_dir: Directory to save output MP4 file
        output_fps: Optional FPS to force in output video
        full_validation: If True, performs thorough validation of output file

    Returns:
        ConversionResult object with status and output path

    Notes:
        - Requires ffmpeg with videotoolbox support
        - Optimized for Apple Silicon hardware
        - Uses h264 high profile with 5Mbps target bitrate
        - Preserves original frame rate from source unless output_fps is specified
        - Removes audio track
        - Uses full color range
    """
    start_time = time.time()
    _, input_wmv_path = participant_file_paths
    input_path = _ensure_path(input_wmv_path)

    output_path_str = mp4_output_path(participant_id, output_dir)
    output_path = _ensure_path(output_path_str)

    logger.info(f"Converting {input_path} to {output_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if output_path.exists():
        logger.info(f"MP4 file already exists: {output_path}")

        # Validate existing file
        is_valid, error = _validate_video(
            output_path,
            source_path=input_path if full_validation else None,
            full_check=full_validation,
        )

        if is_valid:
            logger.info("Existing MP4 file is valid")
            return ConversionResult(
                status=ConversionStatus.ALREADY_EXISTS,
                output_str=output_path.as_posix(),
            )

        logger.warning(f"Existing MP4 is invalid: {error}")

    # Perform conversion
    logger.info("Converting WMV to MP4")
    try:
        # Build ffmpeg command
        cmd = build_ffmpeg_command(
            input_path.as_posix(), output_path.as_posix(), output_fps=output_fps
        )

        cmd_str = " ".join(cmd)
        logger.debug(f"Running command: {cmd_str}")

        result = subprocess.run(
            cmd,
            capture_output=True,  # Capture output for logging
            text=True,
            bufsize=1,  # Line buffered output
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            logger.error(f"FFMPEG Error: {error_msg}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                output_str=output_path.as_posix(),
                error_message=error_msg,
            )

        # Log stdout for debugging
        if result.stdout:
            logger.debug(f"FFMPEG Output: {result.stdout}")

        # Validate converted file
        is_valid, error = _validate_video(
            output_path,
            source_path=input_path if full_validation else None,
            full_check=full_validation,
        )

        if not is_valid:
            return ConversionResult(
                status=ConversionStatus.FAILED,
                output_str=output_path.as_posix(),
                error_message=f"Validation failed: {error}",
            )

        elapsed = time.time() - start_time
        logger.info(f"Conversion completed in {elapsed:.2f}s")

        return ConversionResult(
            status=ConversionStatus.SUCCESS,
            output_str=output_path.as_posix(),
        )

    except Exception as e:
        logger.exception("Exception during MP4 conversion")
        return ConversionResult(
            status=ConversionStatus.FAILED,
            output_str=output_path.as_posix(),
            error_message=str(e),
        )


def output_video_paths(participant_id: str, output_dir: PathLike) -> tuple[str, str]:
    """
    Generate output video paths for chopping and overlaying gaze points.

    Args:
        participant_id: The ID of the participant.
        output_dir: The directory where the output files will be saved.

    Returns:
        Paths for the chopped and overlay videos.
    """
    sanitized_id = participant_id_sanitized(participant_id)
    output_path = _ensure_path(output_dir)
    output_chopped_path = output_path.joinpath(f"{sanitized_id}_chopped.mp4")
    output_overlay_path = output_path.joinpath(f"{sanitized_id}_overlay.mp4")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    if output_chopped_path.exists() and output_overlay_path.exists():
        logger.info(
            f"Chopped and overlayed video files already exist: {output_chopped_path}, {output_overlay_path}"
        )
    else:
        logger.info(
            f"Will create chopped and overlay videos: {output_chopped_path}, {output_overlay_path}"
        )

    return output_chopped_path.as_posix(), output_overlay_path.as_posix()


def iter_video_frames(
    video_path: PathLike,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Generate frames from a video lazily to reduce memory usage.

    Args:
        video_path: Path to video file

    Yields:
        Tuple of (frame_index, frame_data)

    Raises:
        ValueError: If video cannot be opened
    """
    with video_capture(video_path) as video:
        frame_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            yield frame_index, frame
            frame_index += 1


def pyav_timestamps(video_path: PathLike, index: int = 0) -> list[int]:
    """
    Extract timestamps from video using PyAV.

    Args:
        video_path: Video path
        index: Stream index of the video.

    Returns:
        List of timestamps in ms

    Reference:
        https://stackoverflow.com/a/73998721
    """
    video_path = _ensure_path(video_path)
    logger.debug(f"Extracting timestamps from {video_path}")

    container = av.open(video_path)
    video_stream = container.streams.get(index)[0]

    if video_stream.type != "video":
        raise ValueError(
            f"The index {index} is not a video stream. It is a {video_stream.type} stream."
        )

    av_timestamps = []
    for packet in container.demux(video_stream):
        if packet.pts is not None and video_stream.time_base is not None:
            # Safely convert to milliseconds
            timestamp_ms = int(packet.pts * float(video_stream.time_base) * 1000)
            av_timestamps.append(timestamp_ms)

    container.close()
    av_timestamps.sort()

    logger.debug(f"Extracted {len(av_timestamps)} timestamps")
    return av_timestamps


def video_properties(mp4_path: PathLike) -> dict[str, float | int]:
    """Extract video properties using OpenCV.

    Args:
        mp4_path: Path to MP4 file

    Returns:
        Dictionary of video properties
    """
    video_path = _ensure_path(mp4_path)
    logger.debug(f"Getting video properties for {video_path}")

    with video_capture(video_path) as video:
        props = {
            "fps": video.get(cv2.CAP_PROP_FPS),
            "frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": int(video.get(cv2.CAP_PROP_FOURCC)),
        }

    logger.debug(f"Video properties: {props}")
    return props


def gaze_overlay_coords(
    points: pd.DataFrame, current_point_index: int
) -> tuple[int, int]:
    """Extract gaze coordinates for a specific point index.

    Args:
        points: DataFrame with gaze data
        current_point_index: Index of current point

    Returns:
        Tuple of (x, y) coordinates
    """
    row = points.iloc[current_point_index]
    return int(row["Gaze X"]), int(row["Gaze Y"])


def should_skip_frame(
    participant_points: pd.DataFrame, current_point_index: int
) -> bool:
    """Determine if a frame should be skipped based on annotations.

    Args:
        participant_points: DataFrame with participant data
        current_point_index: Index of current point

    Returns:
        True if frame should be skipped, False otherwise
    """
    annotation = participant_points["Respondent Annotations active"].iloc[
        current_point_index
    ]
    return pd.isna(annotation) or annotation == ""


@tag(version="1.0", category="video_processing")
def process_video(
    mp4_conversion: ConversionResult,
    output_video_paths: tuple[str, str],
    participant_points: pd.DataFrame,
    n_frames_proc: int | None = None,
) -> np.ndarray:
    """
    Process video by chopping and adding gaze overlay with existing output file checks.

    Args:
        mp4_conversion: Result of MP4 conversion
        output_video_paths: Tuple of (chopped_path, overlay_path)
        participant_points: DataFrame with participant data
        n_frames_proc: Number of frames to process (None for all)

    Returns:
        NumPy array of gaze data
    """
    start_time = time.time()
    output_chopped_path, output_gazeoverlay_path = output_video_paths
    output_chopped_path = _ensure_path(output_chopped_path)
    output_gazeoverlay_path = _ensure_path(output_gazeoverlay_path)

    # Check if gaze data numpy file already exists
    gaze_npy_path = Path(output_chopped_path).with_suffix(".npy")
    if gaze_npy_path.exists():
        logger.info(f"Found existing gaze data: {gaze_npy_path}")
        try:
            return np.load(gaze_npy_path)
        except Exception as e:
            logger.warning(f"Error loading existing gaze data: {e}")

    # Read the video
    input_video_path = mp4_conversion.output_path
    logger.info(f"Processing video from {input_video_path}")

    # Get video properties
    props = video_properties(input_video_path.as_posix())
    fps = props["fps"]
    total_frames = int(props["frame_count"])
    n_frames_proc = n_frames_proc or total_frames
    width = int(props["width"])
    height = int(props["height"])

    logger.info(
        f"Video properties: {fps} fps, {total_frames} frames, {width}x{height} resolution"
    )
    logger.info(f"Processing {n_frames_proc} frames")

    # Create output video writers if needed
    out_chopping = None
    if output_chopped_path:
        logger.debug(f"Creating chopped video writer: {output_chopped_path}")
        # Use direct integer value for fourcc to avoid attribute error
        # This is equivalent to cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = 0x6D703476  # 'mp4v' in hex
        out_chopping = cv2.VideoWriter(
            output_chopped_path.as_posix(), fourcc, fps, (width, height)
        )

    out_gazeoverlay = None
    if output_gazeoverlay_path:
        logger.debug(f"Creating overlay video writer: {output_gazeoverlay_path}")
        # Use direct integer value for fourcc to avoid attribute error
        # This is equivalent to cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = 0x6D703476  # 'mp4v' in hex
        out_gazeoverlay = cv2.VideoWriter(
            output_gazeoverlay_path.as_posix(), fourcc, fps, (width, height)
        )

    current_point_index = 0
    chopped_frame_index = 0
    skip_frames = False
    gaze_data = []

    # Use lazy frame iteration
    for frame_index, frame in iter_video_frames(input_video_path.as_posix()):
        if frame_index >= n_frames_proc:
            break

        # Calculate current time in milliseconds
        # Avoid division by zero and handle fps safely
        if fps > 0:
            current_time = int(frame_index * 1000 / fps)
        else:
            current_time = 0

        # Find the appropriate point index for the current time
        while (
            current_point_index < len(participant_points) - 1
            and participant_points["Timestamp"].iloc[current_point_index + 1]
            <= current_time
        ):
            current_point_index += 1

            # Check if we should skip this frame based on annotations
            skip_frames = should_skip_frame(participant_points, current_point_index)

        # Process frame if not skipping
        if not skip_frames:
            # Write to chopped video if needed
            if out_chopping is not None:
                out_chopping.write(frame)

            # Add gaze overlay and write to overlay video if needed
            if out_gazeoverlay is not None:
                x, y = gaze_overlay_coords(participant_points, current_point_index)
                overlay_frame = frame.copy()
                cv2.circle(overlay_frame, (x, y), 50, (0, 250, 250), -1)
                out_gazeoverlay.write(overlay_frame)

            # Store gaze data
            gaze_data.append(
                [
                    chopped_frame_index,
                    gaze_overlay_coords(participant_points, current_point_index)[0],
                    gaze_overlay_coords(participant_points, current_point_index)[1],
                    participant_points["Respondent Annotations active"].iloc[
                        current_point_index
                    ],
                ]
            )
            chopped_frame_index += 1

    # Clean up resources
    logger.debug("Releasing video resources")
    if out_chopping is not None:
        out_chopping.release()
    if out_gazeoverlay is not None:
        out_gazeoverlay.release()
    cv2.destroyAllWindows()

    # Log processing statistics
    elapsed = time.time() - start_time
    logger.info(f"Video processing completed in {elapsed:.2f}s")
    logger.info(f"Processed {chopped_frame_index} frames")

    # Return gaze data as numpy array
    if not gaze_data:
        logger.warning("No gaze data collected")
        return np.array([])

    return np.array(gaze_data)


@tag(version="1.0", file_type="numpy")
@datasaver()
def save_gaze_data(
    process_video: np.ndarray,
    output_video_paths: tuple[str, str],
) -> dict:
    """Save gaze data to a numpy file.

    Args:
        process_video: Gaze data from process_video function
        output_video_paths: Tuple of (chopped_path, overlay_path)

    Returns:
        Dictionary with file metadata
    """
    # Get the path to save the gaze data
    gaze_npy_path = Path(output_video_paths[0]).with_suffix(".npy")

    # Save the gaze data
    logger.info(f"Saving gaze data to {gaze_npy_path}")
    np.save(gaze_npy_path, process_video)

    # Return file metadata
    return utils.get_file_metadata(gaze_npy_path)
