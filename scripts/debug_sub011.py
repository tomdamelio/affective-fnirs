#!/usr/bin/env python
"""
Debug script to investigate the trial count issue for sub-011.
Based on debug_trials_issue.py but for sub-011.
"""

import numpy as np
import mne
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data

def main():
    print("=" * 80)
    print("DEBUG: TRIAL COUNT INVESTIGATION FOR SUB-011")
    print("=" * 80)
    
    # Path to sub-011 data - assuming standard BIDS-like structure or from user context
    # User mentioned: sub-011_ses-001_task-fingertapping_desc-validation_report.html
    # File likely in data/raw/sub-011/...
    
    # Need to find the file first.
    base_dir = Path("data/raw/sub-011")
    if not base_dir.exists():
        # Try without 'data/' prefix if running from root and data is elsewhere
         base_dir = Path("c:/Users/tdamelio/Desktop/fnirs/affective-fnirs/data/raw/sub-011")
         
    if not base_dir.exists():
        print(f"Could not find directory: {base_dir}")
        # Identify via pattern match in possible dirs
        import glob
        files = glob.glob("data/raw/sub-011/*.xdf") + glob.glob("c:/Users/tdamelio/Desktop/fnirs/affective-fnirs/data/raw/sub-011/*.xdf")
        if files:
            xdf_path = Path(files[0])
        else:
            print("No XDF file found for sub-011")
            return
    else:
        # Find xdf
        xdf_files = list(base_dir.glob("*.xdf"))
        if not xdf_files:
            print(f"No .xdf files found in {base_dir}")
            return
        xdf_path = xdf_files[0]
        
    print(f"\nLoading XDF file: {xdf_path}")
    streams, header = load_xdf_file(xdf_path)
    identified_streams = identify_streams(streams)
    
    # =========================================================================
    # PART 1: Stream timing comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: STREAM TIMING COMPARISON")
    print("=" * 80)
    
    # EEG stream
    if "eeg" in identified_streams:
        eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
        eeg_start = eeg_timestamps[0]
        eeg_end = eeg_timestamps[-1]
        eeg_duration = eeg_end - eeg_start
        
        print(f"\nEEG Stream:")
        print(f"  Start (LSL): {eeg_start:.3f}s")
        print(f"  End (LSL):   {eeg_end:.3f}s")
        print(f"  Duration:    {eeg_duration:.1f}s ({eeg_duration/60:.1f} min)")
        print(f"  Samples:     {len(eeg_timestamps)}")
        print(f"  Sample rate: {eeg_sfreq} Hz")
    else:
        print("EEG Stream NOT FOUND")
        return
        
    # Marker stream
    if "markers" in identified_streams:
        marker_stream = identified_streams["markers"]
        markers = marker_stream["time_series"]
        marker_timestamps = marker_stream["time_stamps"]
        # Handle markers
        marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
        
        marker_start = marker_timestamps[0]
        marker_end = marker_timestamps[-1]
        marker_duration = marker_end - marker_start
        
        print(f"\nMarker Stream:")
        print(f"  Start (LSL): {marker_start:.3f}s")
        print(f"  End (LSL):   {marker_end:.3f}s")
        print(f"  Duration:    {marker_duration:.1f}s ({marker_duration/60:.1f} min)")
        print(f"  Total markers: {len(markers)}")
    else:
        print("Marker Stream NOT FOUND")
        return
        
    # =========================================================================
    # PART 2: Check coverage
    # =========================================================================
    
    print("\nCheck if markers are within EEG range:")
    valid_count = 0
    total_count = 0
    
    # Assuming config says windows check
    tmin = -3.0 # Default from debug script
    tmax = 15.0
    
    print(f"Using epoch window: [{tmin}, {tmax}]")
    
    for i, (marker, ts) in enumerate(zip(marker_list, marker_timestamps)):
        if marker in ['LEFT', 'RIGHT', 'NOTHING']: # Filter relevant
            epoch_start = ts + tmin
            epoch_end = ts + tmax
            
            is_valid = (epoch_start >= eeg_start) and (epoch_end <= eeg_end)
            status = "VALID" if is_valid else "INVALID (OUT OF RANGE)"
            valid_count += 1 if is_valid else 0
            total_count += 1
            
            print(f"{i}: {marker:<10} ts={ts:.2f} [{epoch_start:.2f}, {epoch_end:.2f}] vs EEG [{eeg_start:.2f}, {eeg_end:.2f}] -> {status}")

    print(f"\nTotal Valid: {valid_count} / {total_count}")
    print(f"Percentage: {valid_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()
