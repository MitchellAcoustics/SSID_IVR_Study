# %%
from hamilton import driver
import ivr_utils
from pathlib import Path

dr = driver.Builder().with_modules(ivr_utils).with_cache().build()

# %%
final_vars = [
    "mp4_conversion",
    "output_video_paths",
    "save_gaze_data",
]

inputs = {
    "participant_id": "P43",
    "n_frames_proc": None,
    "eyetracking_dir": "/Volumes/ritd-ag-project-rd01wq-tober63/SSID IVR Study 1/Eyetracking",
    "output_dir": "/Users/mitch/Documents/UCL/Papers_2025/SSID_IVR_Study/data/output/2025-04-15-test",
}

# %%
dr.visualize_execution(final_vars, "graph.png", inputs=inputs)
dr.execute(
    final_vars,  # type: ignore
    inputs=inputs,
)
run_id = dr.cache.last_run_id
dr.cache.view_run(run_id=run_id, output_file_path="cached_run.png")

# %%
