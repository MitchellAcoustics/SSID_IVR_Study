# %%

import h5py
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple


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
        self, 
        gaze_data: np.ndarray,      
        output_path: Path = None
    ) -> np.ndarray:
        """
        Matches gaze data with segmentation masks.

        Returns: 
            [frame index, x, y, class_id]
        """
        seg_masks = self.get_segmentation_mask()
        num_frames, H, W = seg_masks.shape

        results = []
        for entry in gaze_data:
            frame_index, x, y = entry
            x = int(x)
            y = int(y)
            frame_index = int(frame_index)

            valid = (
                0 <= frame_index < num_frames and
                0 <= x < W and
                0 <= y < H
            )
            class_id = seg_masks[frame_index, y, x] if valid else -1
            results.append([frame_index, x, y, class_id])
    
        result_array = np.array(results, dtype=np.int32)
    
        if output_path:
            np.save(output_path, result_array)
    
        return result_array


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
