"""
CitySeg Auto-Resume Processing Script with H5 Merging
Automatically processes video, checks completion, resumes if interrupted,
and merges all H5 files into a single output file.
"""

import os
import subprocess
import h5py
import numpy as np
import json
import yaml
from pathlib import Path
import sys

# ======================== USER CONFIGURATION ========================

# Input video to process
INPUT_VIDEO = "/Users/yuqiliang/Documents/Github/data/video_processing/P48/P48_chopped_fixedfps.mp4"

# Output directory where H5 files will be saved
OUTPUT_DIR = "/Users/yuqiliang/Documents/Github/data/citgyseg_test/autoresume_test_05/P48"

# Path to CitySeg config.yaml file
CITYSEG_CONFIG = "/Users/yuqiliang/Documents/Github/SSID_IVR_Study/cityseg_configs/config.yaml"

# Maximum retry attempts
MAX_RETRIES = 20

# Batch size for H5 merging (frames per batch)
# Smaller values use less memory but are slower
# Recommended: 20-100
MERGE_BATCH_SIZE = 50

# ====================================================================

# Global variables (do not modify)
TEMP_DIR = None
ALL_H5_FILES = []  # Track all generated H5 files


def verify_paths():
    """Verify that all required paths exist"""
    print("\n" + "=" * 60)
    print("Verifying Configuration")
    print("=" * 60)
    
    # Check input video
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        print("Please update INPUT_VIDEO in the script")
        sys.exit(1)
    print(f"✓ Input video: {INPUT_VIDEO}")
    
    # Check CitySeg config
    if not os.path.exists(CITYSEG_CONFIG):
        print(f"❌ CitySeg config not found: {CITYSEG_CONFIG}")
        print("Please update CITYSEG_CONFIG in the script")
        sys.exit(1)
    print(f"✓ CitySeg config: {CITYSEG_CONFIG}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # Set temp directory
    global TEMP_DIR
    video_dir = Path(INPUT_VIDEO).parent
    TEMP_DIR = str(video_dir / "temp")
    print(f"✓ Temp directory: {TEMP_DIR}")


def update_cityseg_config(input_video, output_dir):
    """Update CitySeg config.yaml with new input and output paths"""
    print("\n" + "=" * 60)
    print("Updating CitySeg Config")
    print("=" * 60)
    
    with open(CITYSEG_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    config['input'] = input_video
    config['output_dir'] = output_dir
    
    with open(CITYSEG_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated: {CITYSEG_CONFIG}")
    print(f"  - input: {input_video}")
    print(f"  - output_dir: {output_dir}")


def find_h5_file(directory):
    """Find the H5 segmentation file in the output directory"""
    h5_files = list(Path(directory).glob("*_segmentation.h5"))
    if h5_files:
        return str(h5_files[0])
    return None


def check_progress(h5_file):
    """
    Check processing progress
    Returns: (processed_frames, total_frames, is_complete)
    """
    if not os.path.exists(h5_file):
        print(f"⚠️  H5 file not found: {h5_file}")
        return 0, None, False
    
    try:
        with h5py.File(h5_file, 'r') as f:
            metadata = json.loads(f['metadata'][()])
            processed_frames = f['segmentation'].shape[0]
            total_frames = metadata['total_video_frames']
            fps = metadata['fps']
        
        is_complete = (processed_frames >= total_frames)
        
        print(f"  Total frames: {total_frames}")
        print(f"  Processed: {processed_frames} frames ({processed_frames/total_frames*100:.1f}%)")
        print(f"  Remaining: {total_frames - processed_frames} frames")
        
        return processed_frames, total_frames, is_complete
        
    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")
        return 0, None, False


def run_cityseg():
    """Run CitySeg processing"""
    print("\n" + "=" * 60)
    print("Running CitySeg Processing")
    print("=" * 60)
    
    # Use absolute path for config
    config_abs_path = str(Path(CITYSEG_CONFIG).absolute())
    
    cmd = ['caffeinate', '-i', 'uv', 'run', 'cityseg', config_abs_path, '--verbose']
    
    print(f"Executing: {' '.join(cmd)}")
    print("Processing started...\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ CitySeg processing completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  CitySeg processing interrupted or failed")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        return False


def extract_remaining_video(original_video, start_frame, output_video):
    """Extract remaining video frames using FFmpeg"""
    print("\n" + "=" * 60)
    print("Extracting Remaining Video Frames")
    print("=" * 60)
    
    # Create temp directory
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', original_video,
        '-vf', f'select=gte(n\\,{start_frame}),setpts=N/FRAME_RATE/TB',
        '-y',
        output_video
    ]
    
    print(f"Extracting from frame {start_frame}...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Video extraction successful: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr}")
        return False


def get_h5_info(h5_file):
    """Get frame information from H5 file"""
    try:
        with h5py.File(h5_file, 'r') as f:
            metadata = json.loads(f['metadata'][()])
            frame_count = f['segmentation'].shape[0]
            return {
                'file': h5_file,
                'frame_count': frame_count,
                'fps': metadata.get('fps', 'N/A'),
                'total_video_frames': metadata.get('total_video_frames', 'N/A')
            }
    except Exception as e:
        return None


def merge_all_h5_files(h5_file_list, output_path, batch_size=50):
    
    print("\n" + "=" * 70)
    print("Merging All H5 Files (Memory-Safe Streaming Mode)")
    print("=" * 70)
    
    if not h5_file_list:
        print("❌ No H5 files to merge")
        return None
    
    if len(h5_file_list) == 1:
        print(f"⚠️  Only one H5 file exists. Copying to output location...")
        import shutil
        shutil.copy2(h5_file_list[0], output_path)
        print(f"✓ Copied: {h5_file_list[0]} -> {output_path}")
        
        # 读取并返回元数据
        try:
            with h5py.File(h5_file_list[0], 'r') as f:
                meta_bytes = f['metadata'][()]
                if isinstance(meta_bytes, bytes):
                    return json.loads(meta_bytes.decode('utf-8'))
                else:
                    return json.loads(meta_bytes)
        except:
            return None
    
    print(f"\nMerging {len(h5_file_list)} H5 files in order:")
    for idx, f in enumerate(h5_file_list, 1):
        file_size = os.path.getsize(f) / (1024**3)
        print(f"  Part {idx}: {os.path.basename(f)} ({file_size:.2f} GB)")
    print(f"\nBatch size: {batch_size} frames per batch")
    
    # ============================================================
    # Step 1: Inspect all files and verify dimensions
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: Inspecting files and verifying dimensions")
    print("=" * 60)
    
    file_info_list = []
    all_metadata = []
    total_frames = 0
    first_shape = None
    first_dtype = None
    
    for idx, h5_path in enumerate(h5_file_list, 1):
        print(f"\nFile {idx}/{len(h5_file_list)}: {os.path.basename(h5_path)}")
        
        try:
            with h5py.File(h5_path, 'r') as f:
   
                seg_shape = f['segmentation'].shape
                seg_dtype = f['segmentation'].dtype
                meta_bytes = f['metadata'][()]
                
                
                if isinstance(meta_bytes, bytes):
                    meta = json.loads(meta_bytes.decode('utf-8'))
                else:
                    meta = json.loads(meta_bytes)
                
                file_info_list.append({
                    'path': h5_path,
                    'shape': seg_shape,
                    'dtype': seg_dtype,
                    'frames': seg_shape[0]
                })
                all_metadata.append(meta)
                
                print(f"  ✓ Frames: {seg_shape[0]}")
                print(f"  ✓ Shape: {seg_shape}")
                print(f"  ✓ Dtype: {seg_dtype}")
                total_frames += seg_shape[0]
                
                if first_shape is None:
                    first_shape = seg_shape[1:]
                    first_dtype = seg_dtype
                elif seg_shape[1:] != first_shape:
                    print(f"❌ Shape mismatch: Expected {first_shape}, got {seg_shape[1:]}")
                    return None
                elif seg_dtype != first_dtype:
                    print(f"⚠️  Warning: Dtype mismatch: Expected {first_dtype}, got {seg_dtype}")
                
        except Exception as e:
            print(f"❌ Error inspecting {h5_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print(f"\n✓ All files verified successfully!")
    print(f"  Total frames to merge: {total_frames}")
    print(f"  Frame shape: {first_shape}")
    print(f"  Data type: {first_dtype}")
    

    base_meta = all_metadata[0]
    merged_metadata = {
        "model_name": base_meta.get("model_name", "unknown"),
        "original_videos": [meta.get("original_video", "unknown") for meta in all_metadata],
        "palette": base_meta.get("palette", []),
        "label_ids": base_meta.get("label_ids", []),
        "frame_step": base_meta.get("frame_step", 1),
        "fps": base_meta.get("fps", 30),
        "total_video_frames": sum(meta.get("total_video_frames", 0) for meta in all_metadata),
        "frame_count": total_frames,
        "merged": True,
        "source_files": [os.path.basename(f['path']) for f in file_info_list],
        "file_frame_counts": [f['frames'] for f in file_info_list]
    }
    
    # ============================================================
    # Step 2: check and create output file with preallocated space
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: Merging data using streaming copy")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print("This may take several minutes for large files...\n")
    
    try:
        # create output H5 file with preallocated space
        with h5py.File(output_path, 'w') as f_out:
            # preallocate space
            full_shape = (total_frames,) + first_shape
            print(f"Allocating space: {full_shape}")
            estimated_size = (np.prod(full_shape) * np.dtype(first_dtype).itemsize / (1024**3))
            
            dset_out = f_out.create_dataset(
                'segmentation',
                shape=full_shape,
                dtype=first_dtype,
                compression='gzip',
                compression_opts=4,
                chunks=True # let h5py choose optimal chunk size
            )
            
            print("✓ Space allocated\n")
            
            # streaming copy data in batches
            current_pos = 0
            
            for idx, info in enumerate(file_info_list, 1):
                print(f"{'─'*60}")
                print(f"Copying Part {idx}/{len(file_info_list)}")
                print(f"  File: {os.path.basename(info['path'])}")
                print(f"  Frame range: {current_pos} → {current_pos + info['frames'] - 1}")
                
                with h5py.File(info['path'], 'r') as f_in:
                    dset_in = f_in['segmentation']
                    
                    # calculate number of batches
                    num_frames = info['frames']
                    num_batches = (num_frames + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_frames)
                        
                        # read batch and write to output
                        batch_data = dset_in[start_idx:end_idx]
                        dset_out[current_pos + start_idx:current_pos + end_idx] = batch_data
                        
                        # progress update
                        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
                            progress = (end_idx / num_frames) * 100
                            print(f"    Progress: {end_idx}/{num_frames} frames ({progress:.1f}%)")
                
                current_pos += info['frames']
                print(f"  ✓ Completed")
            
            # save metadata
            print(f"\n{'─'*60}")
            print("Saving metadata...")
            metadata_json = json.dumps(merged_metadata)
            f_out.create_dataset(
                'metadata',
                data=metadata_json.encode('utf-8')
            )
            print("✓ Metadata saved")
        
        # display file size and stats
        file_size = os.path.getsize(output_path) / (1024**3)  # GB
        
        print(f"\n" + "=" * 60)
        print("Merge Summary")
        print("=" * 60)
        cumulative = 0
        for idx, (info, frames) in enumerate(zip(file_info_list, merged_metadata['file_frame_counts']), 1):
            print(f"Part {idx}: {frames} frames (range: {cumulative}-{cumulative+frames-1})")
            print(f"        {os.path.basename(info['path'])}")
            cumulative += frames
        
        print(f"\n✓ Total merged frames: {total_frames}")
        print(f"✓ Output file: {output_path}")
        print(f"✓ File size: {file_size:.2f} GB")
        print("=" * 60)
        
        return merged_metadata
        
    except Exception as e:
        print(f"\n❌ Error during merge process: {e}")
        import traceback
        traceback.print_exc()
        return None

        # display file size and stats
        file_size = os.path.getsize(output_path) / (1024**3)  # GB
        print(f"\n✓ File saved successfully!")
        print(f"✓ Output file: {output_path}")
        print(f"✓ File size: {file_size:.2f} GB")
        
        # detailed summary
        print(f"\n" + "=" * 60)
        print("Merge Summary")
        print("=" * 60)
        cumulative = 0
        for idx, (info, frames) in enumerate(zip(file_info, merged_metadata['file_frame_counts']), 1):
            print(f"Part {idx}: {frames} frames (range: {cumulative}-{cumulative+frames-1}) - {os.path.basename(info['path'])}")
            cumulative += frames
        print(f"\nTotal merged frames: {total_frames}")
        print("=" * 60)
        
        return merged_metadata
        
    except Exception as e:
        print(f"\n❌ Error during merge process: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main processing loop"""
    print("\n" + "=" * 70)
    print(" " * 8 + "CitySeg Auto-Resume Processing Script with H5 Merging")
    print("=" * 70)
    print(f"Merge batch size: {MERGE_BATCH_SIZE} frames")
    print("=" * 70)
    
    # Verify paths
    verify_paths()
    
    # Update CitySeg config with user-defined paths
    update_cityseg_config(INPUT_VIDEO, OUTPUT_DIR)
    
    iteration = 0
    total_processed_frames = 0
    original_total_frames = None
    
    while iteration < MAX_RETRIES:
        iteration += 1
        
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 70}")
        
        # Determine output directory for this iteration
        if iteration == 1:
            current_output_dir = OUTPUT_DIR
        else:
            current_output_dir = OUTPUT_DIR + f"_part{iteration}"
        
        # Update config for current iteration
        if iteration == 1:
            update_cityseg_config(INPUT_VIDEO, current_output_dir)
        
        # Step 1: Run CitySeg
        success = run_cityseg()
        
        # Step 2: Check progress
        print("\n" + "=" * 60)
        print("Checking Processing Progress")
        print("=" * 60)
        
        h5_file = find_h5_file(current_output_dir)
        
        if not h5_file:
            print("❌ No H5 file found. Processing may have failed.")
            break
        
        # Add to list of all H5 files
        ALL_H5_FILES.append(h5_file)
        
        processed_frames, total_frames, is_complete = check_progress(h5_file)
        
        # Store original total frames
        if iteration == 1:
            original_total_frames = total_frames
        
        # Update cumulative processed frames
        total_processed_frames += processed_frames
        
        # Step 3: Check if complete
        if is_complete or (original_total_frames and total_processed_frames >= original_total_frames):
            print("\n" + "=" * 60)
            print("✅ SUCCESS: Video fully processed!")
            print("=" * 60)
            print(f"  Total frames processed: {total_processed_frames}")
            if original_total_frames:
                print(f"  Original video frames: {original_total_frames}")
            break
        
        # Step 4: If not complete, prepare to process remaining frames
        print("\n" + "=" * 60)
        print("⚠️  Processing incomplete. Preparing to resume...")
        print("=" * 60)
        
        if original_total_frames is None:
            print("❌ Cannot determine total frames. Exiting.")
            break
        
        # Extract remaining video
        remaining_video = os.path.join(TEMP_DIR, f"remaining_part{iteration+1}.mp4")
        if not extract_remaining_video(INPUT_VIDEO, total_processed_frames, remaining_video):
            print("❌ Failed to extract remaining video. Exiting.")
            break
        
        # Update CitySeg config for remaining frames
        remaining_output_dir = OUTPUT_DIR + f"_part{iteration+1}"
        update_cityseg_config(remaining_video, remaining_output_dir)
        
    else:
        print("\n" + "=" * 60)
        print(f"⚠️  Maximum iterations ({MAX_RETRIES}) reached.")
        print("=" * 60)
    
    # Display all generated H5 files
    print("\n" + "=" * 70)
    print("All Generated H5 Files (in chronological order)")
    print("=" * 70)
    print("These files will be merged in this order:")
    print()
    
    cumulative_frames = 0
    for idx, h5_file in enumerate(ALL_H5_FILES, 1):
        info = get_h5_info(h5_file)
        if info:
            print(f"Part {idx}:")
            print(f"  File: {h5_file}")
            print(f"  Frames: {info['frame_count']}")
            print(f"  Frame range: {cumulative_frames} - {cumulative_frames + info['frame_count'] - 1}")
            cumulative_frames += info['frame_count']
    
    print(f"\n{'=' * 70}")
    print(f"Total H5 files generated: {len(ALL_H5_FILES)}")
    print(f"Total frames across all files: {cumulative_frames}")
    if original_total_frames:
        completion_pct = (cumulative_frames / original_total_frames) * 100
        print(f"Completion: {cumulative_frames}/{original_total_frames} ({completion_pct:.1f}%)")
    print("=" * 70)
    
    # Merge all H5 files into one
    if len(ALL_H5_FILES) > 0:
        
        merged_output_path = os.path.join(OUTPUT_DIR, "merged_segmentation.h5")
        print(f"\n{'=' * 70}")
        print("Starting H5 Merge Process")
        print("=" * 70)
        print(f"Files to merge (in order):")
        for idx, h5_file in enumerate(ALL_H5_FILES, 1):
            print(f"  Part {idx}: {os.path.basename(h5_file)}")
        
        merged_metadata = merge_all_h5_files(
            ALL_H5_FILES, 
            merged_output_path, 
            batch_size=MERGE_BATCH_SIZE
        )
        
        if merged_metadata:
            print("\n✅ All H5 files successfully merged!")
            print(f"📁 Merged file location: {merged_output_path}")
            
            # Display merged file info
            file_size = os.path.getsize(merged_output_path) / (1024**3)
            print(f"📊 Total frames in merged file: {merged_metadata['frame_count']}")
            print(f"💾 Merged file size: {file_size:.2f} GB")
        else:
            print("\n⚠️  Merge process completed with warnings (check output above)")
    
    # Final cleanup
    print("\n" + "=" * 60)
    print("Final Cleanup")
    print("=" * 60)
    
    if os.path.exists(TEMP_DIR):
        import shutil
        shutil.rmtree(TEMP_DIR)
        print(f"✓ Deleted temp directory: {TEMP_DIR}")
    
    # Restore original config
    update_cityseg_config(INPUT_VIDEO, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    
    # Final summary
    if ALL_H5_FILES:
        print(f"\n📂 Individual H5 files ({len(ALL_H5_FILES)} parts):")
        for idx, h5_file in enumerate(ALL_H5_FILES, 1):
            print(f"   {idx}. {h5_file}")
        
        merged_file = os.path.join(OUTPUT_DIR, "merged_segmentation.h5")
        if os.path.exists(merged_file):
            print(f"\n🎉 Merged H5 file:")
            print(f"   {merged_file}")
            print(f"\n✨ You can now use the merged file for further analysis!")
        else:
            print(f"\n⚠️  No merged file was created")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        
        # Still display generated H5 files
        if ALL_H5_FILES:
            print("\n" + "=" * 70)
            print("H5 Files Generated Before Interruption")
            print("=" * 70)
            for idx, h5_file in enumerate(ALL_H5_FILES, 1):
                print(f"{idx}. {h5_file}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()