#!/usr/bin/env python
"""
Debug script to investigate ERD/ERS issues and trial count discrepancies.

Issues to investigate:
1. Why are we seeing ERS (synchronization) instead of ERD (desynchronization)?
2. Why only 7 trials per condition instead of 12?
3. Why are effects similar for LEFT and RIGHT in both clusters?
"""

import json
import numpy as np
import mne
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw, embed_events
from affective_fnirs.config import PipelineConfig


def main():
    print("=" * 80)
    print("DEBUG: ERD/ERS ISSUES AND TRIAL COUNT INVESTIGATION")
    print("=" * 80)
    
    # Load data
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    eeg_json_path = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json")
    
    print(f"\n1. Loading XDF file: {xdf_path}")
    
    streams, header = load_xdf_file(xdf_path)
    identified_streams = identify_streams(streams)
    
    # =========================================================================
    # PART 1: Check raw marker stream
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: RAW MARKER STREAM FROM XDF")
    print("=" * 80)
    
    marker_stream = identified_streams["markers"]
    markers = marker_stream["time_series"]
    timestamps = marker_stream["time_stamps"]
    
    print(f"\nTotal markers found: {len(markers)}")
    
    # Count by type
    marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
    
    left_count = sum(1 for m in marker_list if "LEFT" in m.upper())
    right_count = sum(1 for m in marker_list if "RIGHT" in m.upper())
    nothing_count = sum(1 for m in marker_list if "NOTHING" in m.upper())
    
    print(f"\nMarker counts (raw):")
    print(f"  LEFT markers: {left_count}")
    print(f"  RIGHT markers: {right_count}")
    print(f"  NOTHING markers: {nothing_count}")
    
    print(f"\nAll unique markers:")
    unique_markers = set(marker_list)
    for m in sorted(unique_markers):
        count = sum(1 for x in marker_list if x == m)
        print(f"  '{m}': {count}")
    
    print(f"\nFirst 40 markers with timestamps:")
    for i, (m, t) in enumerate(zip(marker_list[:40], timestamps[:40])):
        print(f"  {i:3d}: {t:.3f}s - '{m}'")
    
    # =========================================================================
    # PART 2: Build EEG Raw and check annotations
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: MNE RAW OBJECT AND ANNOTATIONS")
    print("=" * 80)
    
    # Extract EEG data
    eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
    
    # Build EEG Raw
    raw_eeg = build_eeg_raw(
        eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
    )
    
    print(f"\nEEG Raw info:")
    print(f"  Duration: {raw_eeg.times[-1]:.2f}s")
    print(f"  Sampling rate: {raw_eeg.info['sfreq']} Hz")
    print(f"  Channels: {len(raw_eeg.ch_names)}")
    print(f"  First timestamp: {eeg_timestamps[0]:.3f}s")
    print(f"  Last timestamp: {eeg_timestamps[-1]:.3f}s")
    
    # Embed events
    event_mapping = {"LEFT": 1, "RIGHT": 2, "task_start": 10, "task_end": 11}
    raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)
    
    print(f"\nAnnotations in Raw after embed_events:")
    print(f"  Total annotations: {len(raw_eeg.annotations)}")
    
    # Count annotations by description
    ann_counts = {}
    for ann in raw_eeg.annotations:
        desc = ann['description']
        ann_counts[desc] = ann_counts.get(desc, 0) + 1
    
    print(f"\nAnnotation counts:")
    for desc, count in sorted(ann_counts.items()):
        print(f"  '{desc}': {count}")
    
    # Check timing of task annotations
    print(f"\nTask annotation timings:")
    for desc in ['LEFT/1', 'RIGHT/1', 'NOTHING/1']:
        times = [ann['onset'] for ann in raw_eeg.annotations if ann['description'] == desc]
        if times:
            print(f"  {desc}: {len(times)} events")
            print(f"    First: {times[0]:.2f}s, Last: {times[-1]:.2f}s")
            if len(times) > 1:
                intervals = np.diff(times)
                print(f"    Mean interval: {np.mean(intervals):.2f}s")
        else:
            print(f"  {desc}: 0 events")
    
    # =========================================================================
    # PART 3: Check epoch creation with different parameters
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: EPOCH CREATION INVESTIGATION")
    print("=" * 80)
    
    config = PipelineConfig()
    
    print(f"\nEpoch parameters from config:")
    print(f"  tmin: {config.epochs.eeg_tmin_sec}s")
    print(f"  tmax: {config.epochs.eeg_tmax_sec}s")
    print(f"  baseline: ({config.epochs.baseline_tmin_sec}, {config.epochs.baseline_tmax_sec})s")
    
    # Filter EEG first
    raw_filtered = raw_eeg.copy().filter(l_freq=1, h_freq=40, verbose=False)
    
    # Try different event_id configurations
    print("\n--- Testing event_id configurations ---")
    
    # Test 1: Standard event_id
    event_id_1 = {'LEFT/1': 1, 'RIGHT/1': 2, 'NOTHING/1': 3}
    print(f"\nTest 1: event_id = {event_id_1}")
    try:
        events_1, event_id_found_1 = mne.events_from_annotations(raw_filtered, event_id=event_id_1, verbose=False)
        print(f"  Events found: {len(events_1)}")
        for name, eid in event_id_found_1.items():
            count = np.sum(events_1[:, 2] == eid)
            print(f"    {name}: {count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Without /1 suffix
    event_id_2 = {'LEFT': 1, 'RIGHT': 2, 'NOTHING': 3}
    print(f"\nTest 2: event_id = {event_id_2}")
    try:
        events_2, event_id_found_2 = mne.events_from_annotations(raw_filtered, event_id=event_id_2, verbose=False)
        print(f"  Events found: {len(events_2)}")
        for name, eid in event_id_found_2.items():
            count = np.sum(events_2[:, 2] == eid)
            print(f"    {name}: {count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Auto-detect all events
    print(f"\nTest 3: Auto-detect all events (event_id=None)")
    events_3, event_id_found_3 = mne.events_from_annotations(raw_filtered, verbose=False)
    print(f"  Events found: {len(events_3)}")
    print(f"  Event IDs found: {event_id_found_3}")
    for name, eid in event_id_found_3.items():
        count = np.sum(events_3[:, 2] == eid)
        print(f"    {name}: {count}")
    
    # =========================================================================
    # PART 4: Create epochs and check rejection
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: EPOCH CREATION AND REJECTION")
    print("=" * 80)
    
    # Use the event_id that works
    event_id = {'LEFT/1': 1, 'RIGHT/1': 2, 'NOTHING/1': 3}
    events, event_id_found = mne.events_from_annotations(raw_filtered, event_id=event_id, verbose=False)
    
    # Create epochs WITHOUT rejection
    print("\nCreating epochs WITHOUT rejection...")
    epochs_no_reject = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id_found,
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=None,  # No baseline for now
        preload=True,
        verbose=False,
        reject=None,
    )
    
    print(f"  Total epochs (no rejection): {len(epochs_no_reject)}")
    for cond in epochs_no_reject.event_id.keys():
        print(f"    {cond}: {len(epochs_no_reject[cond])}")
    
    # Check which epochs would be dropped due to data limits
    print("\nChecking epoch boundaries vs data limits...")
    for i, event in enumerate(events):
        event_sample = event[0]
        event_time = event_sample / raw_filtered.info['sfreq']
        epoch_start = event_time + config.epochs.eeg_tmin_sec
        epoch_end = event_time + config.epochs.eeg_tmax_sec
        
        event_type = [k for k, v in event_id_found.items() if v == event[2]][0]
        
        in_bounds = epoch_start >= 0 and epoch_end <= raw_filtered.times[-1]
        status = "OK" if in_bounds else "OUT OF BOUNDS"
        
        if not in_bounds or i < 5 or i >= len(events) - 5:
            print(f"  Event {i}: {event_type} at {event_time:.2f}s -> epoch [{epoch_start:.2f}, {epoch_end:.2f}]s - {status}")
    
    # =========================================================================
    # PART 5: Check TFR and baseline correction
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: TFR AND BASELINE CORRECTION")
    print("=" * 80)
    
    # Create epochs with baseline
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id_found,
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
        preload=True,
        verbose=False,
        reject=None,
    )
    
    # Pick C3 and C4
    epochs_motor = epochs.copy().pick_channels(['C3', 'C4'])
    
    freqs = np.arange(8, 30, 1)
    n_cycles = freqs / 2
    
    print(f"\nComputing TFR for C3 and C4...")
    
    # Compute TFR WITHOUT baseline
    tfr_no_bl = mne.time_frequency.tfr_morlet(
        epochs_motor,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        verbose=False,
    )
    
    print(f"TFR shape: {tfr_no_bl.data.shape}")
    print(f"TFR times: {tfr_no_bl.times[0]:.2f} to {tfr_no_bl.times[-1]:.2f}s")
    
    # Define time windows
    baseline_mask = (tfr_no_bl.times >= config.epochs.baseline_tmin_sec) & \
                    (tfr_no_bl.times <= config.epochs.baseline_tmax_sec)
    task_mask = (tfr_no_bl.times >= 0) & (tfr_no_bl.times <= 10)
    
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    print(f"\nRaw power analysis (no baseline correction):")
    for ch_idx, ch_name in enumerate(['C3', 'C4']):
        baseline_alpha = tfr_no_bl.data[ch_idx, alpha_mask, :][:, baseline_mask].mean()
        task_alpha = tfr_no_bl.data[ch_idx, alpha_mask, :][:, task_mask].mean()
        
        baseline_beta = tfr_no_bl.data[ch_idx, beta_mask, :][:, baseline_mask].mean()
        task_beta = tfr_no_bl.data[ch_idx, beta_mask, :][:, task_mask].mean()
        
        alpha_erd = ((task_alpha - baseline_alpha) / baseline_alpha) * 100
        beta_erd = ((task_beta - baseline_beta) / baseline_beta) * 100
        
        print(f"\n  {ch_name}:")
        print(f"    Alpha: baseline={baseline_alpha:.2e}, task={task_alpha:.2e}, ERD={alpha_erd:+.1f}%")
        print(f"    Beta:  baseline={baseline_beta:.2e}, task={task_beta:.2e}, ERD={beta_erd:+.1f}%")
    
    # =========================================================================
    # PART 6: Check by condition
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 6: TFR BY CONDITION (MANUAL CALCULATION)")
    print("=" * 80)
    
    baseline_period = (config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec)
    
    for condition in ['LEFT/1', 'RIGHT/1', 'NOTHING/1']:
        epochs_cond = epochs[condition].copy().pick_channels(['C3', 'C4'])
        
        if len(epochs_cond) == 0:
            print(f"\n{condition}: No epochs!")
            continue
        
        print(f"\n{condition}: {len(epochs_cond)} epochs")
        
        # Compute TFR without baseline
        tfr_cond = mne.time_frequency.tfr_morlet(
            epochs_cond,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False,
        )
        
        # Manual ERD calculation
        for ch_idx, ch_name in enumerate(['C3', 'C4']):
            baseline_alpha = tfr_cond.data[ch_idx, alpha_mask, :][:, baseline_mask].mean()
            task_alpha = tfr_cond.data[ch_idx, alpha_mask, :][:, task_mask].mean()
            
            baseline_beta = tfr_cond.data[ch_idx, beta_mask, :][:, baseline_mask].mean()
            task_beta = tfr_cond.data[ch_idx, beta_mask, :][:, task_mask].mean()
            
            alpha_erd = ((task_alpha - baseline_alpha) / baseline_alpha) * 100
            beta_erd = ((task_beta - baseline_beta) / baseline_beta) * 100
            
            print(f"  {ch_name}: Alpha ERD = {alpha_erd:+.1f}%, Beta ERD = {beta_erd:+.1f}%")
    
    # =========================================================================
    # PART 7: Check the lateralization analysis function
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 7: LATERALIZATION ANALYSIS FUNCTION CHECK")
    print("=" * 80)
    
    from affective_fnirs.lateralization_analysis import compute_lateralization_analysis
    
    print("\nRunning lateralization analysis on filtered data (no CAR)...")
    lat_result = compute_lateralization_analysis(
        raw_filtered,
        alpha_band=(8, 12),
        beta_band=(13, 30),
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
        task_window=(0, 10),
    )
    
    print(f"\nLateralization results:")
    print(f"  Trials: LEFT={lat_result.n_trials_left}, RIGHT={lat_result.n_trials_right}, NOTHING={lat_result.n_trials_nothing}")
    
    print(f"\n  Alpha ERD (%):")
    print(f"    LEFT:    C3={lat_result.left_c3_alpha_erd:+.1f}, C4={lat_result.left_c4_alpha_erd:+.1f}")
    print(f"    RIGHT:   C3={lat_result.right_c3_alpha_erd:+.1f}, C4={lat_result.right_c4_alpha_erd:+.1f}")
    print(f"    NOTHING: C3={lat_result.nothing_c3_alpha_erd:+.1f}, C4={lat_result.nothing_c4_alpha_erd:+.1f}")
    
    print(f"\n  Beta ERD (%):")
    print(f"    LEFT:    C3={lat_result.left_c3_beta_erd:+.1f}, C4={lat_result.left_c4_beta_erd:+.1f}")
    print(f"    RIGHT:   C3={lat_result.right_c3_beta_erd:+.1f}, C4={lat_result.right_c4_beta_erd:+.1f}")
    print(f"    NOTHING: C3={lat_result.nothing_c3_beta_erd:+.1f}, C4={lat_result.nothing_c4_beta_erd:+.1f}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
