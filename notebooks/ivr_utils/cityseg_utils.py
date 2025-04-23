# %%

import h5py
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple
import pandas as pd


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
            [frame_index, x, y, session id ,class_id]
        """
        seg_masks = self.get_segmentation_mask()
        num_frames, H, W = seg_masks.shape

        results = []
        for entry in gaze_data:
            frame_index, x, y , session_id = entry
            x = int(x)
            y = int(y)
            frame_index = int(frame_index)

            valid = (
                0 <= frame_index < num_frames and
                0 <= x < W and
                0 <= y < H
            )
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
            'MiradorSanNicolas1': '0QUE',
            'BidderSt2': '1V',
            'BidderSt1': '28V',
            'BlairSt1': '22V',
            'BlundellSt1': '24V',
            'CaledonianPark1': '26V',
            'CamdenTown4': '2V',
            'CarloV2': '16V',
            'DadongSq3': '25V',
            'EustonTap3': '15V',
            'HereEspresso1': '7V',
            'IvesRd2': '17V',
            'KingsfordRow1': '21V',
            'LianhuashanParkEntrance1': '27V',
            'MarchmontGardens4': '23V',
            'NewRiverWalk1': '3V',
            'OlympicSq3': '19V',
            'PancrasLock2': '20V',
            'PingshanSt1': '6V',
            'PlazaBibRambla1': '5V',
            'RegentsParkFields2': '13V',
            'RegentsParkJapan2': '8V',
            'RiverLeeSchools1b': '9V',
            'SanMarco1': '11V',
            'StephensonSt1': '10V',
            'TateModern3': '4V',
            'TorringtonSq4': '12V',
            'WineOfficeCt1': '14V',
            'ZhongshanPark5': '18V',
            'Blank': '29V'
        }

        scene_dict = {}
        for row in matched_data:
            session_id = row[3]
            class_id = row[4]
            if session_id not in scene_dict:
                scene_dict[session_id] = {"total": 0, "counts": {}}
            scene_dict[session_id]["total"] += 1
            scene_dict[session_id]["counts"][class_id] = scene_dict[session_id]["counts"].get(class_id, 0) + 1

        
        output_rows = []
        for session_id, data in scene_dict.items():
            total = data["total"]
            scene_id = session_scene_mapping.get(session_id, 'Unknown')
            for cid in range(1, 19):
                count = data["counts"].get(cid, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                percentage = f"{percentage:.2f}%"
                output_rows.append([scene_id, session_id, cid, percentage])

        
        result_array = np.array(output_rows, dtype=object)
        print(result_array.shape) 
        return result_array

    

    def merge_percentage_with_circumplex(self, circumplex_path: Path, percentage_data: np.array, PARTICIPANT, output_file=None) -> pd.DataFrame:

        percentage_df = pd.DataFrame(
            percentage_data,
            columns=["SceneID", "SessionID", "ClassID", "Percentage"]
        )
        percentage_df["Percentage"] = percentage_df["Percentage"].str.replace("%", "").astype(float)

        pivot_df = percentage_df.pivot_table(
            index="SessionID",
            columns="ClassID",
            values="Percentage",
            aggfunc="first"
        )
        pivot_df.columns = [f"Class{int(col)}" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        av_df = pd.read_excel(circumplex_path, sheet_name="AV")
        av_df = av_df[av_df["Participant"] == PARTICIPANT].copy()
        av_df = av_df[['Participant', 'SessionID', 'ISOPleasant', 'ISOEventful']]

        merged = pd.merge(
            av_df,
            pivot_df,
            on="SessionID",
            how="left"
        )

        class_cols = sorted(
            [col for col in merged.columns if col.startswith("Class")],
            key=lambda x: int(x.replace("Class", ""))
        )
        desired_columns = ["Participant", "SessionID", "ISOPleasant", "ISOEventful"] + class_cols
        merged = merged[desired_columns]

        if output_file is not None:
            suffix = output_file.suffix.lower()
            if suffix == ".csv":
                merged.to_csv(output_file, index=False)
            elif suffix in [".xls", ".xlsx"]:
                with pd.ExcelWriter(output_file) as writer:
                    merged.to_excel(writer, index=False)
            else:
                merged.to_csv(output_file, index=False)


        return merged





    
  




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
