#!/usr/bin/env python
"""
Debug script v2 - Comprehensive investigation of ERD/ERS issues.

FINDINGS FROM V1:
1. EEG recording is INCOMPLETE - only 660s of ~1050s experiment
2. Event names are INCONSISTENT: LEFT/1, RIGHT/2, NOTHING (no suffix)
3. Only 22/36 events are within EEG time range
4. Only 7 LEFT events are usable (others are RIGHT/2 or NOTHING)

This script will:
- Fix event naming to be consistent
- Analyze all available trials correctly
- Verify ERD calculations are correct
"""

import json
import numpy as np
import mne
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw
from affective_fnirs.config import PipelineConfig


def main():
    print("=" * 80)
    print("DEBUG V2: COMPREHENSIVE ERD/ERS INVESTIGATION")
    print("=" * 80)
    
    # Load data
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    
    print(f"\n1. Loading XDF file: {xdf_path}")
    
    streams, header = load_xdf_file(xdf_path)
    identified_streams = identify_streams(streams)
    
    # =========================================================================
    # PART 1: Understand the data timing issue
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: DATA TIMING ANALYSIS")
    print("=" * 80)
    
    # Get marker stream
    marker_stream = identified_streams["markers"]
    markers = marker_stream["time_series"]
    marker_timestamps = marker_stream["time_stamps"]
    
    marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
    
    # Get EEG stream
    eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
    
    print(f"\nMarker stream:")
    print(f"  Total markers: {len(markers)}")
    print(f"  First marker time: {marker_timestamps[0]:.3f}s")
    print(f"  Last marker time: {marker_timestamps[-1]:.3f}s")
    print(f"  Duration: {marker_timestamps[-1] - marker_timestamps[0]:.1f}s")
    
    print(f"\nEEG stream:")
    print(f"  First sample time: {eeg_timestamps[0]:.3f}s")
    print(f"  Last sample time: {eeg_timestamps[-1]:.3f}s")
    print(f"  Duration: {eeg_timestamps[-1] - eeg_timestamps[0]:.1f}s")
    
    # Count events within EEG range
    events_in_range = sum(1 for t in marker_timestamps 
                         if eeg_timestamps[0] <= t <= eeg_timestamps[-1])
    print(f"\nEvents within EEG range: {events_in_range}/{len(markers)}")
    
    # Count by type within range
    left_in_range = sum(1 for m, t in zip(marker_list, marker_timestamps) 
                        if "LEFT" in m.upper() and eeg_timestamps[0] <= t <= eeg_timestamps[-1])
    right_in_range = sum(1 for m, t in zip(marker_list, marker_timestamps) 
                         if "RIGHT" in m.upper() and eeg_timestamps[0] <= t <= eeg_timestamps[-1])
    nothing_in_range = sum(1 for m, t in zip(marker_list, marker_timestamps) 
                           if "NOTHING" in m.upper() and eeg_timestamps[0] <= t <= eeg_timestamps[-1])
    
    print(f"\nEvents by type within EEG range:")
    print(f"  LEFT: {left_in_range}")
    print(f"  RIGHT: {right_in_range}")
    print(f"  NOTHING: {nothing_in_range}")
    
    # =========================================================================
    # PART 2: Build EEG Raw with CORRECT event embedding
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: BUILD EEG RAW WITH CORRECT EVENTS")
    print("=" * 80)
    
    # Build EEG Raw
    raw_eeg = build_eeg_raw(
        eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
    )
    
    # Manually embed events with CONSISTENT naming
    recording_start_lsl = eeg_timestamps[0]
    recording_end_lsl = eeg_timestamps[-1]
    
    onsets = []
    descriptions = []
    
    for marker_time, marker_name in zip(marker_timestamps, marker_list):
        # Skip events outside EEG range
        if marker_time < recording_start_lsl or marker_time > recording_end_lsl:
            continue
        
        onset_sec = marker_time - recording_start_lsl
        onsets.append(onset_sec)
        descriptions.append(marker_name)  # Keep original name: LEFT, RIGHT, NOTHING
    
    # Create annotations
    annotations = mne.Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=descriptions,
    )
    raw_eeg.set_annotations(annotations)
    
    print(f"\nAnnotations embedded:")
    ann_counts = {}
    for ann in raw_eeg.annotations:
        desc = ann['description']
        ann_counts[desc] = ann_counts.get(desc, 0) + 1
    
    for desc, count in sorted(ann_counts.items()):
        print(f"  '{desc}': {count}")
    
    # =========================================================================
    # PART 3: Create epochs with correct event IDs
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: EPOCH CREATION")
    print("=" * 80)
    
    config = PipelineConfig()
    
    # Filter EEG
    raw_filtered = raw_eeg.copy().filter(l_freq=1, h_freq=40, verbose=False)
    
    # Create events with correct IDs
    event_id = {'LEFT': 1, 'RIGHT': 2, 'NOTHING': 3}
    events, event_id_found = mne.events_from_annotations(raw_filtered, event_id=event_id, verbose=False)
    
    print(f"\nEvents found:")
    for name, eid in event_id_found.items():
        count = np.sum(events[:, 2] == eid)
        print(f"  {name}: {count}")
    
    # Create epochs
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id_found,
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=None,  # No baseline yet - we'll apply it to TFR
        preload=True,
        verbose=False,
        reject=None,
    )
    
    print(f"\nEpochs created:")
    for cond in epochs.event_id.keys():
        print(f"  {cond}: {len(epochs[cond])}")
    
    # =========================================================================
    # PART 4: Compute TFR and ERD correctly
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: TFR AND ERD CALCULATION")
    print("=" * 80)
    
    freqs = np.arange(8, 30, 1)
    n_cycles = freqs / 2
    
    baseline_period = (config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec)
    
    # Define time masks
    baseline_tmin, baseline_tmax = baseline_period
    task_tmin, task_tmax = 0, 10  # Task window
    
    alpha_freqs = (8, 12)
    beta_freqs = (13, 30)
    
    print(f"\nParameters:")
    print(f"  Baseline: {baseline_period}")
    print(f"  Task window: ({task_tmin}, {task_tmax})")
    print(f"  Alpha: {alpha_freqs} Hz")
    print(f"  Beta: {beta_freqs} Hz")
    
    # Compute ERD for each condition
    print("\n" + "-" * 40)
    print("ERD RESULTS (Manual calculation)")
    print("-" * 40)
    
    results = {}
    
    for condition in ['LEFT', 'RIGHT', 'NOTHING']:
        if condition not in epochs.event_id:
            print(f"\n{condition}: No epochs!")
            continue
        
        epochs_cond = epochs[condition].copy().pick_channels(['C3', 'C4'])
        n_epochs = len(epochs_cond)
        
        if n_epochs == 0:
            print(f"\n{condition}: No epochs!")
            continue
        
        print(f"\n{condition}: {n_epochs} epochs")
        
        # Compute TFR without baseline
        tfr = mne.time_frequency.tfr_morlet(
            epochs_cond,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False,
        )
        
        # Get time and frequency masks
        times = tfr.times
        baseline_mask = (times >= baseline_tmin) & (times <= baseline_tmax)
        task_mask = (times >= task_tmin) & (times <= task_tmax)
        
        alpha_mask = (freqs >= alpha_freqs[0]) & (freqs <= alpha_freqs[1])
        beta_mask = (freqs >= beta_freqs[0]) & (freqs <= beta_freqs[1])
        
        results[condition] = {}
        
        for ch_idx, ch_name in enumerate(['C3', 'C4']):
            # Get power data
            power = tfr.data[ch_idx]  # (n_freqs, n_times)
            
            # Calculate ERD manually: (task - baseline) / baseline * 100
            baseline_alpha = power[alpha_mask, :][:, baseline_mask].mean()
            task_alpha = power[alpha_mask, :][:, task_mask].mean()
            alpha_erd = ((task_alpha - baseline_alpha) / baseline_alpha) * 100
            
            baseline_beta = power[beta_mask, :][:, baseline_mask].mean()
            task_beta = power[beta_mask, :][:, task_mask].mean()
            beta_erd = ((task_beta - baseline_beta) / baseline_beta) * 100
            
            results[condition][ch_name] = {
                'alpha_erd': alpha_erd,
                'beta_erd': beta_erd,
                'baseline_alpha': baseline_alpha,
                'task_alpha': task_alpha,
                'baseline_beta': baseline_beta,
                'task_beta': task_beta,
            }
            
            print(f"  {ch_name}: Alpha ERD = {alpha_erd:+.1f}%, Beta ERD = {beta_erd:+.1f}%")
    
    # =========================================================================
    # PART 5: Summary and interpretation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: SUMMARY AND INTERPRETATION")
    print("=" * 80)
    
    print("\n--- Expected patterns (motor imagery/execution) ---")
    print("LEFT hand movement → ERD in RIGHT motor cortex (C4)")
    print("RIGHT hand movement → ERD in LEFT motor cortex (C3)")
    print("NOTHING → No significant ERD")
    
    print("\n--- Observed patterns ---")
    if 'LEFT' in results and 'RIGHT' in results:
        print(f"\nLEFT condition:")
        print(f"  C3 (ipsilateral): Alpha={results['LEFT']['C3']['alpha_erd']:+.1f}%, Beta={results['LEFT']['C3']['beta_erd']:+.1f}%")
        print(f"  C4 (contralateral): Alpha={results['LEFT']['C4']['alpha_erd']:+.1f}%, Beta={results['LEFT']['C4']['beta_erd']:+.1f}%")
        
        print(f"\nRIGHT condition:")
        print(f"  C3 (contralateral): Alpha={results['RIGHT']['C3']['alpha_erd']:+.1f}%, Beta={results['RIGHT']['C3']['beta_erd']:+.1f}%")
        print(f"  C4 (ipsilateral): Alpha={results['RIGHT']['C4']['alpha_erd']:+.1f}%, Beta={results['RIGHT']['C4']['beta_erd']:+.1f}%")
        
        if 'NOTHING' in results:
            print(f"\nNOTHING condition (control):")
            print(f"  C3: Alpha={results['NOTHING']['C3']['alpha_erd']:+.1f}%, Beta={results['NOTHING']['C3']['beta_erd']:+.1f}%")
            print(f"  C4: Alpha={results['NOTHING']['C4']['alpha_erd']:+.1f}%, Beta={results['NOTHING']['C4']['beta_erd']:+.1f}%")
        
        # Check if patterns are correct
        print("\n--- Validation ---")
        left_c4_erd = results['LEFT']['C4']['alpha_erd'] < -10  # Expect ERD in C4 for LEFT
        right_c3_erd = results['RIGHT']['C3']['alpha_erd'] < -10  # Expect ERD in C3 for RIGHT
        
        print(f"LEFT shows C4 ERD (expected): {'✓' if left_c4_erd else '✗'}")
        print(f"RIGHT shows C3 ERD (expected): {'✓' if right_c3_erd else '✗'}")
    
    # =========================================================================
    # PART 6: Plot spectrograms for visual inspection
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 6: GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for row_idx, condition in enumerate(['LEFT', 'RIGHT', 'NOTHING']):
        if condition not in epochs.event_id:
            continue
        
        epochs_cond = epochs[condition].copy().pick_channels(['C3', 'C4'])
        
        if len(epochs_cond) == 0:
            continue
        
        tfr = mne.time_frequency.tfr_morlet(
            epochs_cond,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False,
        )
        
        # Apply percent baseline
        tfr.apply_baseline(baseline=baseline_period, mode='percent', verbose=False)
        
        for col_idx, ch_name in enumerate(['C3', 'C4']):
            ax = axes[row_idx, col_idx]
            ch_idx = tfr.ch_names.index(ch_name)
            
            # Plot spectrogram (multiply by 100 for percentage)
            im = ax.imshow(
                tfr.data[ch_idx] * 100,
                aspect='auto',
                origin='lower',
                extent=[tfr.times[0], tfr.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-50,
                vmax=50,
            )
            
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.axvline(10, color='black', linestyle='--', linewidth=1)
            ax.axhline(8, color='white', linestyle=':', alpha=0.5)
            ax.axhline(12, color='white', linestyle=':', alpha=0.5)
            ax.axhline(13, color='white', linestyle=':', alpha=0.5)
            ax.axhline(30, color='white', linestyle=':', alpha=0.5)
            
            ax.set_title(f"{condition} - {ch_name} ({len(epochs_cond)} trials)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            
            plt.colorbar(im, ax=ax, label="ERD/ERS (%)")
    
    plt.tight_layout()
    plt.savefig("outputs/debug_erd_spectrograms.png", dpi=150)
    print(f"\nSaved diagnostic plot to: outputs/debug_erd_spectrograms.png")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
