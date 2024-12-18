from pathlib import Path


def find_participant_files(
    participant_id: str, data_dir: Path | str
) -> tuple[Path, Path]:
    """
    Find the participant files for the given participant ID in the given data directory.
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
