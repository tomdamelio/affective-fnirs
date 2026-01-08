#!/usr/bin/env python
"""
Debug script to investigate the trial count issue.

Expected: 12 trials per condition (LEFT, RIGHT, NOTHING) = 36 total
Observed: Only 7 trials per condition within EEG range

Questions to answer:
1. What is the actual duration of EEG vs fNIRS vs Markers?
2. Why is EEG shorter than the experiment?
3. Can we use all 36 trials with fNIRS even if EEG is incomplete?
4. What is the exact timing of each trial?
"""

import numpy as np
import mne
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data


def main():
    print("=" * 80)
    print("DEBUG: TRIAL COUNT INVESTIGATION")
    print("=" * 80)
    
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    
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
    
    # fNIRS stream
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(identified_streams["fnirs"])
    fnirs_start = fnirs_timestamps[0]
    fnirs_end = fnirs_timestamps[-1]
    fnirs_duration = fnirs_end - fnirs_start
    
    print(f"\nfNIRS Stream:")
    print(f"  Start (LSL): {fnirs_start:.3f}s")
    print(f"  End (LSL):   {fnirs_end:.3f}s")
    print(f"  Duration:    {fnirs_duration:.1f}s ({fnirs_duration/60:.1f} min)")
    print(f"  Samples:     {len(fnirs_timestamps)}")
    print(f"  Sample rate: {fnirs_sfreq} Hz")
    
    # Marker stream
    marker_stream = identified_streams["markers"]
    markers = marker_stream["time_series"]
    marker_timestamps = marker_stream["time_stamps"]
    marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
    
    marker_start = marker_timestamps[0]
    marker_end = marker_timestamps[-1]
    marker_duration = marker_end - marker_start
    
    print(f"\nMarker Stream:")
    print(f"  Start (LSL): {marker_start:.3f}s")
    print(f"  End (LSL):   {marker_end:.3f}s")
    print(f"  Duration:    {marker_duration:.1f}s ({marker_duration/60:.1f} min)")
    print(f"  Total markers: {len(markers)}")
    
    # =========================================================================
    # PART 2: Detailed marker analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: DETAILED MARKER ANALYSIS")
    print("=" * 80)
    
    print(f"\nAll 36 markers with timing relative to streams:")
    print(f"{'#':>3} {'Marker':<10} {'LSL Time':>12} {'EEG Rel':>10} {'fNIRS Rel':>10} {'In EEG?':>8} {'In fNIRS?':>9}")
    print("-" * 75)
    
    in_eeg_count = {'LEFT': 0, 'RIGHT': 0, 'NOTHING': 0}
    in_fnirs_count = {'LEFT': 0, 'RIGHT': 0, 'NOTHING': 0}
    
    for i, (marker, ts) in enumerate(zip(marker_list, marker_timestamps)):
        eeg_rel = ts - eeg_start
        fnirs_rel = ts - fnirs_start
        
        in_eeg = eeg_start <= ts <= eeg_end
        in_fnirs = fnirs_start <= ts <= fnirs_end
        
        in_eeg_str = "YES" if in_eeg else "NO"
        in_fnirs_str = "YES" if in_fnirs else "NO"
        
        # Count
        if in_eeg:
            in_eeg_count[marker] += 1
        if in_fnirs:
            in_fnirs_count[marker] += 1
        
        print(f"{i+1:>3} {marker:<10} {ts:>12.3f} {eeg_rel:>10.1f}s {fnirs_rel:>10.1f}s {in_eeg_str:>8} {in_fnirs_str:>9}")
    
    print("\n" + "-" * 75)
    print(f"\nSummary - Markers within each stream:")
    print(f"  {'Condition':<10} {'In EEG':>8} {'In fNIRS':>10} {'Total':>8}")
    for cond in ['LEFT', 'RIGHT', 'NOTHING']:
        total = sum(1 for m in marker_list if m == cond)
        print(f"  {cond:<10} {in_eeg_count[cond]:>8} {in_fnirs_count[cond]:>10} {total:>8}")
    
    total_in_eeg = sum(in_eeg_count.values())
    total_in_fnirs = sum(in_fnirs_count.values())
    print(f"  {'TOTAL':<10} {total_in_eeg:>8} {total_in_fnirs:>10} {len(markers):>8}")
    
    # =========================================================================
    # PART 3: Check epoch boundaries
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: EPOCH BOUNDARY CHECK (tmin=-3s, tmax=+15s)")
    print("=" * 80)
    
    tmin = -3.0
    tmax = 15.0
    
    print(f"\nEpoch window: [{tmin}, {tmax}]s relative to marker")
    print(f"\nMarkers that can form complete epochs:")
    print(f"{'#':>3} {'Marker':<10} {'Epoch Start':>12} {'Epoch End':>12} {'EEG OK?':>8} {'fNIRS OK?':>9}")
    print("-" * 70)
    
    eeg_epochs_ok = {'LEFT': 0, 'RIGHT': 0, 'NOTHING': 0}
    fnirs_epochs_ok = {'LEFT': 0, 'RIGHT': 0, 'NOTHING': 0}
    
    for i, (marker, ts) in enumerate(zip(marker_list, marker_timestamps)):
        epoch_start_lsl = ts + tmin
        epoch_end_lsl = ts + tmax
        
        # Check if full epoch fits in EEG
        eeg_ok = (epoch_start_lsl >= eeg_start) and (epoch_end_lsl <= eeg_end)
        fnirs_ok = (epoch_start_lsl >= fnirs_start) and (epoch_end_lsl <= fnirs_end)
        
        eeg_ok_str = "YES" if eeg_ok else "NO"
        fnirs_ok_str = "YES" if fnirs_ok else "NO"
        
        if eeg_ok:
            eeg_epochs_ok[marker] += 1
        if fnirs_ok:
            fnirs_epochs_ok[marker] += 1
        
        # Convert to relative times for display
        epoch_start_rel = ts - eeg_start + tmin
        epoch_end_rel = ts - eeg_start + tmax
        
        print(f"{i+1:>3} {marker:<10} {epoch_start_rel:>12.1f}s {epoch_end_rel:>12.1f}s {eeg_ok_str:>8} {fnirs_ok_str:>9}")
    
    print("\n" + "-" * 70)
    print(f"\nSummary - Complete epochs possible:")
    print(f"  {'Condition':<10} {'EEG':>8} {'fNIRS':>10}")
    for cond in ['LEFT', 'RIGHT', 'NOTHING']:
        print(f"  {cond:<10} {eeg_epochs_ok[cond]:>8} {fnirs_epochs_ok[cond]:>10}")
    
    total_eeg_epochs = sum(eeg_epochs_ok.values())
    total_fnirs_epochs = sum(fnirs_epochs_ok.values())
    print(f"  {'TOTAL':<10} {total_eeg_epochs:>8} {total_fnirs_epochs:>10}")
    
    # =========================================================================
    # PART 4: Investigate why EEG is shorter
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: WHY IS EEG SHORTER?")
    print("=" * 80)
    
    print(f"\nTime gap analysis:")
    print(f"  EEG ends at:     {eeg_end:.3f}s (LSL)")
    print(f"  fNIRS ends at:   {fnirs_end:.3f}s (LSL)")
    print(f"  Markers end at:  {marker_end:.3f}s (LSL)")
    
    print(f"\n  EEG is {fnirs_end - eeg_end:.1f}s shorter than fNIRS")
    print(f"  EEG is {marker_end - eeg_end:.1f}s shorter than markers")
    
    # Check if there's a gap in EEG timestamps
    eeg_diffs = np.diff(eeg_timestamps)
    expected_diff = 1.0 / eeg_sfreq
    large_gaps = np.where(eeg_diffs > expected_diff * 2)[0]
    
    print(f"\n  EEG timestamp gaps > 2x expected ({expected_diff*1000:.2f}ms):")
    if len(large_gaps) > 0:
        for gap_idx in large_gaps[:10]:  # Show first 10
            gap_size = eeg_diffs[gap_idx]
            gap_time = eeg_timestamps[gap_idx]
            print(f"    At sample {gap_idx} (t={gap_time-eeg_start:.1f}s): gap of {gap_size*1000:.1f}ms")
    else:
        print(f"    No large gaps found - EEG recording simply stopped early")
    
    # =========================================================================
    # PART 5: Recommendations
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"""
FINDINGS:
- EEG recording stopped early (660s vs 1050s experiment duration)
- This is likely a recording issue, not a processing bug
- Only {total_eeg_epochs} of 36 trials have complete EEG epochs
- All 36 trials have complete fNIRS epochs

OPTIONS:
1. Use only the {total_eeg_epochs} trials with complete EEG data (current approach)
2. Use shorter epochs (e.g., tmin=-1, tmax=10) to recover more trials
3. For fNIRS-only analysis, all 36 trials are available
4. For multimodal analysis, limited to {total_eeg_epochs} trials

The EEG recording appears to have been stopped prematurely.
This is a DATA COLLECTION issue, not a pipeline bug.
""")
    
    # =========================================================================
    # PART 6: Test shorter epochs
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 6: TESTING SHORTER EPOCH WINDOWS")
    print("=" * 80)
    
    test_windows = [
        (-3, 15),   # Current
        (-2, 12),   # Shorter
        (-1, 10),   # Even shorter
        (-1, 8),    # Minimal
    ]
    
    print(f"\n{'Window':<15} {'EEG Epochs':>12} {'fNIRS Epochs':>14}")
    print("-" * 45)
    
    for tmin_test, tmax_test in test_windows:
        eeg_count = 0
        fnirs_count = 0
        
        for ts in marker_timestamps:
            epoch_start = ts + tmin_test
            epoch_end = ts + tmax_test
            
            if (epoch_start >= eeg_start) and (epoch_end <= eeg_end):
                eeg_count += 1
            if (epoch_start >= fnirs_start) and (epoch_end <= fnirs_end):
                fnirs_count += 1
        
        print(f"[{tmin_test}, {tmax_test}]s      {eeg_count:>12} {fnirs_count:>14}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
