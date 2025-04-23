# %%

import h5py
import json
import logging
import datetime
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple, Union

import cityseg as cs
from hamilton.function_modifiers import tag, cache, datasaver, save_to, source
from hamilton.io import utils
from hamilton.caching import fingerprinting

# Import custom fingerprinting functions for h5py objects
from ivr_utils.fingerprinting import hash_cityseg_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]


def _load_hdf_file(file_path: Path) -> Tuple[h5py.File, Dict[str, Any]]:
    """
    Loads segmentation data and metadata from an HDF file.

    Args:
        file_path (Path): Path to the HDF file.

    Returns:
        Tuple[h5py.File, Dict[str, Any]]: Loaded HDF file and metadata.
    """
    hdf_file = h5py.File(file_path, "r")
    json_metadata = hdf_file["metadata"][()]
    metadata = json.loads(json_metadata)
    if "palette" in metadata and isinstance(metadata["palette"], list):
        metadata["palette"] = np.array(metadata["palette"], np.uint8)
    return hdf_file, metadata


@dataclass
class CitySegData:
    """
    Dataclass for CitySeg data.
    """

    hdf_file: h5py.File
    metadata: Dict[str, Any]
    hdf_path: Path

    @classmethod
    def from_hdf(cls, hdf_path: Path) -> "CitySegData":
        """
        Loads CitySeg data from an HDF file.

        Args:
            hdf_path (Path): Path to the HDF file.

        Returns:
            CitySegData: CitySeg data.
        """
        hdf_file, metadata = _load_hdf_file(hdf_path)
        return CitySegData(hdf_file, metadata, hdf_path)

    @property
    def palette(self) -> np.ndarray:
        """
        Returns the palette of the segmentation data.

        Returns:
            np.ndarray: Palette of the segmentation data.
        """
        return self.metadata["palette"]

    def get_segmentation_mask(self) -> np.ndarray:
        """
        Returns the segmentation mask.

        Returns:
            np.ndarray: Segmentation mask.
        """
        return self.hdf_file["segmentation"][()]

    def match_gaze_with_masks(
        self, gaze_data: np.ndarray, output_path: Path = None
    ) -> np.ndarray:
        """
        Matches gaze data with segmentation masks.

        Returns:
            [frame_index, x, y, session id ,class_id]
        """
        seg_masks = self.get_segmentation_mask()
        num_frames, H, W = seg_masks.shape

        results = []
        for entry in gaze_data:
            frame_index, x, y, session_id = entry
            x = int(x)
            y = int(y)
            frame_index = int(frame_index)

            valid = 0 <= frame_index < num_frames and 0 <= x < W and 0 <= y < H
            class_id = seg_masks[frame_index, y, x] if valid else -1
            results.append([frame_index, x, y, session_id, class_id])
            result_array = np.array(results, dtype=object)

        if output_path:
            np.save(output_path, result_array)

        return result_array

    def calculate_percentage_of_mask(self, gaze_data: np.ndarray) -> np.ndarray:
        """
        Calculate the percentage of the class id for each scene.

        Returns:
            Percentage of the class id for each scene for each participant.
        """
        matched_data = self.match_gaze_with_masks(gaze_data, None)

        # session id scene mapping
        session_scene_mapping = {
            "MiradorSanNicolas1": "0QUE",
            "BidderSt2": "1V",
            "BidderSt1": "28V",
            "BlairSt1": "22V",
            "BlundellSt1": "24V",
            "CaledonianPark1": "26V",
            "CamdenTown4": "2V",
            "CarloV2": "16V",
            "DadongSq3": "25V",
            "EustonTap3": "15V",
            "HereEspresso1": "7V",
            "IvesRd2": "17V",
            "KingsfordRow1": "21V",
            "LianhuashanParkEntrance1": "27V",
            "MarchmontGardens4": "23V",
            "NewRiverWalk1": "3V",
            "OlympicSq3": "19V",
            "PancrasLock2": "20V",
            "PingshanSt1": "6V",
            "PlazaBibRambla1": "5V",
            "RegentsParkFields2": "13V",
            "RegentsParkJapan2": "8V",
            "RiverLeeSchools1b": "9V",
            "SanMarco1": "11V",
            "StephensonSt1": "10V",
            "TateModern3": "4V",
            "TorringtonSq4": "12V",
            "WineOfficeCt1": "14V",
            "ZhongshanPark5": "18V",
            "Blank": "29V",
        }

        scene_dict = {}
        for row in matched_data:
            session_id = row[3]
            class_id = row[4]
            if session_id not in scene_dict:
                scene_dict[session_id] = {"total": 0, "counts": {}}
            scene_dict[session_id]["total"] += 1
            scene_dict[session_id]["counts"][class_id] = (
                scene_dict[session_id]["counts"].get(class_id, 0) + 1
            )

        output_lines = []
        for session_id, data in scene_dict.items():
            total = data["total"]
            scene_id = session_scene_mapping.get(session_id, "Unknown")
            for cid in range(1, 19):
                count = data["counts"].get(cid, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                line = f"{scene_id} {session_id} {cid} {percentage:.0f}%"
                output_lines.append(line)

        return np.array(output_lines, dtype=object)


# Register the hash function for CitySegData
fingerprinting.hash_value.register(CitySegData, hash_cityseg_data)


# Hamilton functions for CitySeg processing


def _ensure_path(path: PathLike) -> Path:
    """Convert string path to Path object if needed.

    Args:
        path: Path as string or Path object

    Returns:
        Path object
    """
    return Path(path) if isinstance(path, str) else path


@tag(category="cityseg_configuration")
def cityseg_config(cityseg_config_path: str) -> cs.Config:
    """Load the CitySeg configuration from a YAML file.

    This function creates the base configuration object from the specified YAML file.

    Args:
        cityseg_config_path: Path to the CitySeg configuration YAML file

    Returns:
        Loaded CitySeg configuration object
    """
    logger.info(f"Loading CitySeg configuration from {cityseg_config_path}")
    config = cs.Config.from_yaml(cityseg_config_path)
    return config


@tag(category="cityseg_configuration")
@cache(behavior="disable")  # Ignore caching for Config objects
def cityseg_prepared_config(
    cityseg_config: cs.Config,
    process_video: np.ndarray,
    output_video_paths: tuple[str, str],
    participant_id: str,
    frame_step: int = 15,
    batch_size: int = 5,
    device: str = "mps",
) -> cs.Config:
    """Prepare the CitySeg configuration for processing a specific video.

    Configures the CitySeg object with the appropriate paths, processing parameters,
    and device settings for the specified participant's video.

    Args:
        cityseg_config: Base CitySeg configuration
        process_video: Video processing result containing frame data
        output_video_paths: Tuple of paths to chopped and overlay videos
        participant_id: Participant identifier
        frame_step: Process every Nth frame (higher values = faster but less detailed)
        batch_size: Number of frames to process in each batch
        device: Computing device ('mps' for M2 Mac, 'cuda' for NVIDIA, 'cpu' for fallback)

    Returns:
        Fully configured CitySeg configuration object
    """
    config = cityseg_config

    # Get the chopped video path
    chopped_video_path = _ensure_path(output_video_paths[0])

    logger.info(f"Configuring CitySeg for video: {chopped_video_path}")

    # Update configuration for this specific video
    config.input = chopped_video_path
    config.frame_step = frame_step
    config.batch_size = batch_size
    config.model.device = device
    config.input_type = config._determine_input_type()

    # Set output directory based on participant ID
    seg_output_dir = chopped_video_path.parent / f"{participant_id}_cityseg_output"
    config.output_dir = seg_output_dir

    logger.info(f"CitySeg output directory: {seg_output_dir}")
    logger.info(
        f"CitySeg frame step: {frame_step}, batch size: {batch_size}, device: {device}"
    )

    return config


@tag(category="cityseg_processing", version="1.0")
@cache(behavior="disable")  # Ignore caching for Path objects
def cityseg_segmentation_output_directory(cityseg_prepared_config: cs.Config) -> Path:
    """Run CitySeg segmentation and return the output directory.

    Processes the video with CitySeg to generate semantic segmentation.
    This is the most computationally expensive step and is therefore cached.

    Args:
        cityseg_prepared_config: Fully configured CitySeg configuration

    Returns:
        Path to the directory containing the segmentation results
    """
    config = cityseg_prepared_config

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if segmentation has already been completed
    segmentation_file = list(output_dir.glob("*_segmentation.h5"))
    if segmentation_file and not config.force_reprocess:
        logger.info(f"Using existing segmentation at {segmentation_file[0]}")
        return output_dir

    # Run segmentation
    logger.info(f"Running CitySeg on {config.input}")
    processor = cs.create_processor(config)
    processor.process()
    logger.info(f"CitySeg processing completed. Results in {output_dir}")

    return output_dir


@tag(category="cityseg_processing")
@cache(behavior="disable")  # Now using our custom hash function for CitySegData
def cityseg_segmentation_data(
    cityseg_segmentation_output_directory: Path,
) -> CitySegData:
    """Load the CitySeg segmentation data from the output directory.

    Finds and loads the HDF5 segmentation file produced by CitySeg.
    This function uses our custom hash function for CitySegData to enable caching.

    Args:
        cityseg_segmentation_output_directory: Path to the segmentation output

    Returns:
        CitySegData object containing the segmentation data and metadata
    """
    output_dir = cityseg_segmentation_output_directory

    # Find segmentation file
    segmentation_files = list(output_dir.glob("*_segmentation.h5"))
    assert len(segmentation_files) > 0, f"No segmentation file found in {output_dir}"

    logger.info(f"Loading segmentation data from {segmentation_files[0]}")

    # Load segmentation data
    return CitySegData.from_hdf(segmentation_files[0])


@tag(category="cityseg_analysis", version="1.0")
@cache(behavior="disable")  # Ignore caching for CitySegData objects
@datasaver()
def cityseg_save_matched_gaze(
    cityseg_segmentation_data: CitySegData,
    save_gaze_data: dict,
    participant_id: str,
    output_dir: str,  # Changed to str to avoid Path caching issues
) -> dict:
    """Match gaze coordinates with segmentation masks and save the results.

    For each gaze point, identifies the semantic segment class at that position.
    This is useful for understanding what scene elements participants looked at.
    The results are saved to {output_dir}/cityseg_analysis/{participant_id}_matched_gaze.npy.

    Args:
        cityseg_segmentation_data: Loaded segmentation data
        save_gaze_data: Metadata from saved gaze data
        participant_id: Participant identifier
        output_dir: Base output directory

    Returns:
        Dictionary with metadata about the saved file
    """
    seg_data = cityseg_segmentation_data

    # Get gaze data path and load it
    gaze_data_path = Path(save_gaze_data["file_metadata"]["path"])

    # Check if gaze data exists
    if not gaze_data_path.exists():
        raise FileNotFoundError(f"Gaze data not found at {gaze_data_path}")

    logger.info(f"Loading gaze data from {gaze_data_path}")
    gaze_data = np.load(gaze_data_path)

    # Create analysis directory
    output_dir_path = _ensure_path(output_dir)
    analysis_dir = output_dir_path / "cityseg_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Match gaze with segments
    logger.info(f"Matching gaze data with segmentation masks for {participant_id}")
    matched_data = seg_data.match_gaze_with_masks(gaze_data, None)
    logger.info(f"Matched {len(matched_data)} gaze points with segmentation masks")

    # Save matched gaze data
    matched_path = analysis_dir / f"{participant_id}_matched_gaze.npy"
    np.save(matched_path, matched_data)
    logger.info(f"Saved matched gaze data to {matched_path}")

    # Return metadata
    return {
        "path": matched_path.as_posix(),
        "format": "npy",
        "shape": matched_data.shape,
        "participant_id": participant_id,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@tag(category="cityseg_analysis", version="1.0")
@cache(behavior="disable")  # Ignore caching for CitySegData objects
@datasaver()
def cityseg_save_percentages(
    cityseg_segmentation_data: CitySegData,
    save_gaze_data: dict,
    participant_id: str,
    output_dir: str,
) -> dict:
    """Calculate and save the percentage of gaze points falling in each segment class.

    Computes what percentage of participant attention was directed at
    different semantic elements (road, sidewalk, buildings, etc.).
    The results are saved to:
    - {output_dir}/cityseg_analysis/{participant_id}_class_percentages.npy (binary)
    - {output_dir}/cityseg_analysis/{participant_id}_class_percentages.csv (text)

    Args:
        cityseg_segmentation_data: Loaded segmentation data
        save_gaze_data: Metadata from saved gaze data
        participant_id: Participant identifier
        output_dir: Base output directory

    Returns:
        Dictionary with metadata about the saved files
    """
    seg_data = cityseg_segmentation_data

    # Get gaze data path and load it
    gaze_data_path = Path(save_gaze_data["file_metadata"]["path"])
    logger.info(f"Loading gaze data from {gaze_data_path} for percentage calculation")
    gaze_data = np.load(gaze_data_path)

    # Create analysis directory
    output_dir_path = _ensure_path(output_dir)
    analysis_dir = output_dir_path / "cityseg_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Calculate percentages
    logger.info("Calculating segment class percentages")
    percentages = seg_data.calculate_percentage_of_mask(gaze_data)
    logger.info(f"Calculated percentages for {len(percentages)} segment classes")

    # Save as NPY file
    percentages_path = analysis_dir / f"{participant_id}_class_percentages.npy"
    np.save(percentages_path, percentages)
    logger.info(f"Saved class percentages to {percentages_path}")

    # Save as CSV file
    percentages_csv_path = analysis_dir / f"{participant_id}_class_percentages.csv"
    with open(percentages_csv_path, "w") as f:
        for line in percentages:
            f.write(f"{line}\n")
    logger.info(f"Saved class percentages CSV to {percentages_csv_path}")

    # Return metadata
    return {
        "npy_path": percentages_path.as_posix(),
        "csv_path": percentages_csv_path.as_posix(),
        "format": ["npy", "csv"],
        "shape": percentages.shape,
        "participant_id": participant_id,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@tag(category="cityseg_analysis", version="1.0")
@datasaver()
def cityseg_save_analysis_results(
    cityseg_save_matched_gaze: dict,
    cityseg_save_percentages: dict,
    participant_id: str,
    output_dir: str,
) -> dict:
    """Consolidate metadata about saved CitySeg analysis results.

    This function serves as a terminal node for the CitySeg processing pipeline,
    collecting metadata about all saved files from the individual save operations.
    The actual saving is performed by the individual datasaver functions.

    Args:
        cityseg_save_matched_gaze: Metadata from saved matched gaze data
        cityseg_save_percentages: Metadata from saved percentages data (includes both NPY and CSV)
        participant_id: Participant identifier
        output_dir: Base output directory

    Returns:
        Dictionary with consolidated metadata about all saved files
    """
    output_dir_path = _ensure_path(output_dir)
    analysis_dir = output_dir_path / "cityseg_analysis"

    logger.info(f"CitySeg analysis results saved to {analysis_dir}")
    logger.info(f"All CitySeg analysis outputs for {participant_id} have been saved")

    # Consolidate metadata
    return {
        "matched_gaze_path": cityseg_save_matched_gaze["path"],
        "percentages_path": cityseg_save_percentages["npy_path"],
        "percentages_csv_path": cityseg_save_percentages["csv_path"],
        "participant_id": participant_id,
        "analysis_directory": str(analysis_dir),
        "timestamp": datetime.datetime.now().isoformat(),
        "files_saved": [
            cityseg_save_matched_gaze["path"],
            cityseg_save_percentages["npy_path"],
            cityseg_save_percentages["csv_path"],
        ],
    }


# %%
# Usage:

if __name__ == "__main__":
    # Load CitySeg data from an HDF file
    cityseg_test = Path(
        "/Users/mitch/Documents/UCL/Papers_2025/SSID_IVR_Study/data/outputs/segmentations/mask2former-swin-large-cityscapes-semantic_step15/output_chopped_P7_fixedfps_mask2former-swin-large-cityscapes-semantic_step15_segmentation.h5"
    )
    cityseg_data = CitySegData.from_hdf(cityseg_test)

    # Get the palette of the segmentation data
    palette = cityseg_data.palette
