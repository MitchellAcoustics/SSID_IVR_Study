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
    - Python 3.9+
    - macOS with Apple Silicon (for hardware acceleration)

Example:
    >>> from ivr_utils import find_participant_files
    >>> csv_path, video_path = find_participant_files("P001", "data/")
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import cv2  # For video processing and validation
import math  # For frame count comparison
import subprocess  # For ffmpeg operations
import logging  # For operation logging
from typing import Union, Optional, Tuple, List
import av  # For advanced video operations
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def find_participant_files(
    participant_id: str, data_dir: Path | str
) -> tuple[Path, Path]:
    """
    Find the participant files for the given participant ID in the given data directory.

    This function searches for two specific files associated with a participant:
    1. A CSV file containing eyetracking data.
    2. A WMV video file located in the "Gaze Replays" subdirectory.

    The function ensures that exactly one CSV file and one WMV video file are found.
    If either file is not found or if multiple files are found, an assertion error is raised.

    Parameters:
    - participant_id (str): The ID of the participant whose files are to be found.
    - data_dir (Path | str): The directory where the participant files are located. This can be a Path object or a string.

    Returns:
    - tuple[Path, Path]: A tuple containing the resolved paths to the participant's CSV file and WMV video file.

    Raises:
    - AssertionError: If the CSV file or the WMV video file is not found or if multiple files are found.
    """
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    # Find eyetracking csv
    part_csv_l = list(data_dir.rglob(f"*{participant_id}.csv"))
    assert len(part_csv_l) == 1, "Participant CSV not found or more than one found"

    part_csv_path = part_csv_l[0]
    part_csv_path = part_csv_path.resolve()

    # Find eyetracking video
    scenario_vid_dir = part_csv_path.parent / "Gaze Replays"
    part_vid_l = list(scenario_vid_dir.rglob(f"*{participant_id}*.wmv"))
    assert len(part_vid_l) == 1, "Participant video not found or more than one found"

    part_vid_path = part_vid_l[0]
    part_vid_path = part_vid_path.resolve()

    return part_csv_path, part_vid_path


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
    status: ConversionStatus
    output_path: Path
    error_message: Optional[str] = None

    def print_status(self):
        if self.status == ConversionStatus.SUCCESS:
            print("Conversion successful")
        elif self.status == ConversionStatus.ALREADY_EXISTS:
            print("MP4 file already exists and is valid")
        elif self.status == ConversionStatus.FAILED:
            print("Conversion failed")


@dataclass
class VideoInfo:
    frame_count: int
    fps: float
    duration: float


def _ensure_path(path: Union[str, Path]) -> Path:
    return Path(path) if isinstance(path, str) else path


def _get_video_info(video_path: Path) -> Optional[VideoInfo]:
    """Extract frame count, FPS and duration from video file"""
    cap = cv2.VideoCapture(video_path.as_posix())
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    cap.release()

    return VideoInfo(frame_count, fps, duration)


def _validate_video(
    video_path: Path, source_path: Optional[Path] = None, full_check: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Validates video file integrity and optionally compares it with a source video.

    Parameters:
    - video_path (Path): The path to the video file to be validated.
    - source_path (Optional[Path]): The path to the source video file for comparison (optional).
    - full_check (bool): If True, performs a full check including comparison with the source video.

    Returns:
    - tuple[bool, Optional[str]]: A tuple containing a boolean indicating if the video is valid,
                                and an optional error message if the video is not valid.
    """
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

    return True, None


def convert_wmv_to_mp4(
    input_wmv_path: Union[Path, str],
    output_mp4_path: Optional[Union[Path, str]] = None,
    full_validation: bool = True,
) -> ConversionResult:
    """Convert WMV video to MP4 format using Apple Silicon hardware acceleration.

    This function converts WMV videos to MP4 format using the h264_videotoolbox codec
    for hardware acceleration on Apple Silicon. It includes validation of the output
    file and handles existing files intelligently.

    Args:
        input_wmv_path: Path to input WMV file. Can be string or Path object.
        output_mp4_path: Optional path for output MP4 file. If None, creates MP4
            in same location as input with .mp4 extension.
        full_validation: If True, performs thorough validation of output file including
            frame count and duration comparison with source (default: True).

    Returns:
        ConversionResult object with fields:
            - status: ConversionStatus enum (SUCCESS, FAILED, ALREADY_EXISTS)
            - output_path: Path to output MP4 file
            - error_message: String description of error if failed

    Raises:
        No direct exceptions, all errors are captured in ConversionResult

    Examples:
        Basic conversion:
        >>> result = convert_wmv_to_mp4("input.wmv")
        >>> if result.status == ConversionStatus.SUCCESS:
        >>>     print(f"Converted to {result.output_path}")

        Custom output path with validation:
        >>> result = convert_wmv_to_mp4(
        >>>     "input.wmv",
        >>>     "output/video.mp4",
        >>>     full_validation=True
        >>> )

    Notes:
        - Requires ffmpeg with videotoolbox support
        - Optimized for Apple Silicon hardware
        - Uses h264 high profile with 5Mbps target bitrate
        - Preserves original frame rate from source
        - Removes audio track
        - Uses full color range
    """
    logger = logging.getLogger(__name__)

    input_path = _ensure_path(input_wmv_path)
    output_path = (
        _ensure_path(output_mp4_path)
        if output_mp4_path
        else input_path.with_suffix(".mp4")
    )

    # Check existing file
    if output_path.exists():
        logger.info("Checking existing MP4 file")
        is_valid, error = _validate_video(
            output_path,
            source_path=input_path if full_validation else None,
            full_check=full_validation,
        )
        if is_valid:
            return ConversionResult(
                status=ConversionStatus.ALREADY_EXISTS, output_path=output_path
            )
        logger.warning(f"Existing MP4 is invalid: {error}")

    # Perform conversion
    logger.info("Converting WMV to MP4")
    try:
        # fmt: off
        result = subprocess.run(
            [
                "ffmpeg",

                # Hardware acceleration for Apple Silicon
                "-hwaccel", "videotoolbox", 

                "-i", input_path.as_posix(),

                # Video encoding settings
                "-c:v", "h264_videotoolbox",
                "-b:v", "5M",          # Target bitrate of 5 Mbps
                "-maxrate", "7M",      # Maximum bitrate
                "-bufsize", "10M",     # Buffer size
                
                "-profile:v", "high",  # High profile for better quality
                "-movflags", "+faststart",  # Enable streaming optimization

                # Color settings
                "-color_range", "1",   # Full color range
                
                # To force 30 fps - much slower, unsure if needed
                # "-fps_mode", "cfr",
                # "-r", "30",

                "-an",  # no audio
                "-y",  # overwrite output file
                "-stats",  # show progress
                "-v", "warning",  # show warnings
                output_path.as_posix(),
            ],
            capture_output=False,  # Let stats print to console
            text=True,
            bufsize=1,  # Line buffered output
        )
        # fmt: on

        if result.returncode != 0:
            logger.error(f"FFMPEG Error: {result.stderr}")
            return ConversionResult(
                status=ConversionStatus.FAILED,
                output_path=output_path,
                error_message=result.stderr,
            )

        if result.returncode == 0:
            # Validate converted file
            is_valid, error = _validate_video(
                output_path,
                source_path=input_path if full_validation else None,
                full_check=full_validation,
            )
            if not is_valid:
                return ConversionResult(
                    status=ConversionStatus.FAILED,
                    output_path=output_path,
                    error_message=f"Validation failed: {error}",
                )

            return ConversionResult(
                status=ConversionStatus.SUCCESS, output_path=output_path
            )

    except Exception as e:
        return ConversionResult(
            status=ConversionStatus.FAILED,
            output_path=output_path,
            error_message=str(e),
        )


def pyav_timestamps(video: Path, index: int = 0) -> List[int]:
    """
    https://stackoverflow.com/a/73998721
    Parameters:
        video (str): Video path
        index (int): Stream index of the video.
    Returns:
        List of timestamps in ms
    """
    container = av.open(video)
    video = container.streams.get(index)[0]

    if video.type != "video":
        raise ValueError(
            f"The index {index} is not a video stream. It is an {video.type} stream."
        )

    av_timestamps = [
        int(packet.pts * video.time_base * 1000)
        for packet in container.demux(video)
        if packet.pts is not None
    ]

    container.close()
    av_timestamps.sort()

    return av_timestamps


def process_video_by_annotations(
    annotation_data: pd.DataFrame,
    input_video_path: Path,
    output_video_path: Optional[Path] = None,
    max_duration_seconds: Optional[float] = None,
) -> Path:
    """Process a video by removing segments without valid annotations.

    This function creates a new video that only includes segments where annotations are present
    and valid. It uses PyAV for efficient video processing and matches video frames to annotations
    using precise timestamp synchronization rather than frame counting.

    The function maintains temporal consistency by adjusting frame presentation timestamps (PTS)
    to account for removed segments, ensuring smooth playback of the output video.

    Args:
        annotation_data: DataFrame containing annotation data with required columns:
            - "Timestamp": Time in milliseconds when annotation occurred
            - "Respondent Annotations active": Boolean or non-null value indicating valid annotation
        input_video_path: Path to the source video file to process
        output_video_path: Optional path for the processed video. If not provided,
            appends "_processed" to the input filename
        max_duration_seconds: Optional maximum duration to process in seconds.
            If None, processes the entire video

    Returns:
        Path: Path to the processed output video file

    Technical Details:
        - Preserves original video codec parameters (resolution, pixel format)
        - Uses h264 codec for output with yuv420p pixel format
        - Maintains frame rate and aspect ratio from source
        - Adjusts PTS values to account for removed segments
        - Uses binary search for efficient timestamp matching
        - Shows progress with time-based updates

    Example:
        >>> annotation_df = pd.read_csv("annotations.csv")
        >>> input_video = Path("raw_video.mp4")
        >>> processed_video = process_video_by_annotations(
        ...     annotation_df,
        ...     input_video,
        ...     max_duration_seconds=300  # Process first 5 minutes
        ... )
    """
    if output_video_path is None:
        output_video_path = input_video_path.with_name(
            f"{input_video_path.stem}_chopped{input_video_path.suffix}"
        )

    # Initialize video containers
    input_container = av.open(input_video_path)
    try:
        # Get input video stream and metadata
        input_stream = input_container.streams.video[0]

        # Calculate frame timing parameters
        fps = float(input_stream.average_rate)
        frame_interval_seconds = 1 / fps
        frame_interval_timebase = int(frame_interval_seconds / input_stream.time_base)

        # Get video dimensions
        frame_width = input_stream.width
        frame_height = input_stream.height

        # Calculate processing duration in milliseconds
        if max_duration_seconds is None:
            processing_duration_ms = float(
                input_stream.duration * input_stream.time_base * 1000
            )
        else:
            processing_duration_ms = max_duration_seconds * 1000

        # Setup output video container with matching parameters
        output_container = av.open(output_video_path, mode="w")
        output_stream = output_container.add_stream("h264")
        output_stream.time_base = input_stream.time_base
        output_stream.width = frame_width
        output_stream.height = frame_height
        output_stream.pix_fmt = "yuv420p"  # Standard pixel format for compatibility

        # Initialize frame tracking variables
        previous_frame_pts = 0  # Previous frame's presentation timestamp
        last_output_pts = 0  # Last written frame's presentation timestamp
        is_previous_frame_skipped = False
        processed_frame_count = 0

        # Setup progress bar
        progress_bar = tqdm(
            total=int(processing_duration_ms / 1000),
            bar_format="{l_bar}{bar}| {n:.0f} of {total_fmt} s [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
            unit="s",
        )

        # Process each frame
        for frame in input_container.decode(video=0):
            original_pts = frame.pts

            # Convert frame timestamp to milliseconds
            current_time_ms = float(frame.pts * input_stream.time_base * 1000)

            # Check if we've reached the maximum processing duration
            if current_time_ms >= processing_duration_ms:
                break

            # Find the closest annotation timestamp using binary search
            annotation_index = np.searchsorted(
                annotation_data["Timestamp"], current_time_ms
            )

            # Check if there's a valid annotation at this time
            has_valid_annotation = annotation_index < len(
                annotation_data
            ) and not pd.isna(
                annotation_data["Respondent Annotations active"].iloc[annotation_index]
            )

            if has_valid_annotation:
                # Calculate new presentation timestamp accounting for removed segments
                pts_adjustment = (
                    original_pts - previous_frame_pts
                    if not is_previous_frame_skipped
                    else frame_interval_timebase
                )
                new_pts = last_output_pts + pts_adjustment

                # Update frame timestamp and encode
                frame.pts = new_pts
                encoded_packet = output_stream.encode(frame)
                if encoded_packet:
                    output_container.mux(encoded_packet)

                # Update tracking variables for next frame
                is_previous_frame_skipped = False
                last_output_pts = new_pts
            else:
                is_previous_frame_skipped = True

            # Update progress bar
            output_time_seconds = float(last_output_pts * input_stream.time_base)
            progress_bar.set_description(
                f"Output time: {output_time_seconds:.2f}s | "
                f"{'Skipping' if is_previous_frame_skipped else 'Processing'}"
            )

            time_delta = (current_time_ms / 1000) - float(
                previous_frame_pts * input_stream.time_base
            )
            progress_bar.update(time_delta)

            # Update tracking variables
            processed_frame_count += 1
            previous_frame_pts = original_pts

        # Complete progress bar
        progress_bar.n = progress_bar.total
        progress_bar.refresh()

    finally:
        # Clean up resources
        input_container.close()
        output_container.close()
        progress_bar.close()

    return output_video_path


def overlaying_video(input_video_path, output_overlay_path, time_proc, points):
    """Overlay gaze points on video using PyAV.

    This function reads a video file using PyAV and creates a new video with gaze points
    overlaid as circles. The matchup between video frames and gaze points is based on
    the actual timestamps embedded in the video rather than frame count calculations.

    Args:
        input_video_path: Path to input video file
        output_overlay_path: Path for output video with overlays
        time_proc: Maximum time to process in seconds
        points: DataFrame containing timestamps and gaze coordinates

    Returns:
        Path to output video file with overlays
    """
    import numpy as np

    # Open input video
    input_container = av.open(str(input_video_path))
    input_stream = input_container.streams.video[0]

    # Get video metadata
    fps = float(input_stream.average_rate)
    width = input_stream.width
    height = input_stream.height

    # Create output video container
    output_container = av.open(str(output_overlay_path), mode="w")
    output_stream = output_container.add_stream("h264", rate=fps)
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = "yuv420p"

    current_point_index = 0
    current_time = 0

    # Process frames
    for frame in input_container.decode(video=0):
        if current_time >= time_proc * 1000:
            break

        # Get timestamp in milliseconds
        current_time = int(frame.pts * input_stream.time_base * 1000)

        # Update point index based on timestamp
        while (
            current_point_index < len(points) - 1
            and points["Timestamp"].iloc[current_point_index + 1] <= current_time
        ):
            current_point_index += 1

        # Convert frame to numpy array for drawing
        img = frame.to_ndarray(format="bgr24")

        # Draw gaze point if available
        if current_point_index < len(points):
            row = points.iloc[current_point_index]
            x, y = int(row["Gaze X"]), int(row["Gaze Y"])
            cv2.circle(img, (x, y), 50, (0, 250, 250), -1)

        # Convert back to PyAV frame and encode
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        packet = output_stream.encode(new_frame)
        if packet:
            output_container.mux(packet)

        frames_processed += 1

    # Flush remaining frames
    packet = output_stream.encode(None)
    if packet:
        output_container.mux(packet)

    # Close containers
    input_container.close()
    output_container.close()

    return output_overlay_path
