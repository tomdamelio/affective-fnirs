#!/usr/bin/env python
"""
Debug why only 7 epochs are being created when 21 should be available.
"""

import numpy as np
import mne
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw, embed_events
from affective_fnirs.config import PipelineConfig


def main():
    print("=" * 80)
    print("DEBUG: EPOCH CREATION ISSUE")
    print("=" * 80)
    
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    
    streams, header = load_xdf_file(xdf_path)
    identified_streams = identify_streams(streams)
    
    # Extract data
    eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
    marker_stream = identified_streams["markers"]
    
    # Build EEG Raw
    raw_eeg = build_eeg_raw(
        eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
    )
    
    print(f"\nEEG Raw duration: {raw_eeg.times[-1]:.1f}s")
    
    # =========================================================================
    # Test 1: Embed events with current mapping
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Current event mapping")
    print("=" * 80)
    
    event_mapping = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
        "task_start": 10,
        "task_end": 11,
    }
    
    raw_test1 = raw_eeg.copy()
    raw_test1 = embed_events(raw_test1, marker_stream, event_mapping)
    
    print(f"\nAnnotations after embed_events:")
    ann_counts = {}
    for ann in raw_test1.annotations:
        desc = ann['description']
        ann_counts[desc] = ann_counts.get(desc, 0) + 1
    
    for desc, count in sorted(ann_counts.items()):
        print(f"  '{desc}': {count}")
    
    # Check annotation timings
    print(f"\nAnnotation timings:")
    for desc in ['LEFT/1', 'RIGHT/2', 'NOTHING/3']:
        times = [ann['onset'] for ann in raw_test1.annotations if ann['description'] == desc]
        if times:
            print(f"  {desc}: {len(times)} events, range [{min(times):.1f}, {max(times):.1f}]s")
    
    # =========================================================================
    # Test 2: Create epochs with verbose output
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Epoch creation")
    print("=" * 80)
    
    config = PipelineConfig()
    raw_filtered = raw_test1.copy().filter(l_freq=1, h_freq=40, verbose=False)
    
    # Get events
    event_id = {'LEFT/1': 1, 'RIGHT/2': 2, 'NOTHING/3': 3}
    
    print(f"\nTrying event_id: {event_id}")
    events, event_id_found = mne.events_from_annotations(raw_filtered, event_id=event_id, verbose=True)
    
    print(f"\nEvents found: {len(events)}")
    print(f"Event ID mapping: {event_id_found}")
    
    for name, eid in event_id_found.items():
        count = np.sum(events[:, 2] == eid)
        print(f"  {name}: {count}")
    
    # Create epochs
    print(f"\nCreating epochs with tmin={config.epochs.eeg_tmin_sec}, tmax={config.epochs.eeg_tmax_sec}")
    
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=event_id_found,
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=None,
        preload=True,
        verbose=True,
        reject=None,
    )
    
    print(f"\nEpochs created: {len(epochs)}")
    for cond in epochs.event_id.keys():
        print(f"  {cond}: {len(epochs[cond])}")
    
    # Check drop log
    print(f"\nDrop log analysis:")
    dropped = [i for i, log in enumerate(epochs.drop_log) if log]
    kept = [i for i, log in enumerate(epochs.drop_log) if not log]
    print(f"  Kept: {len(kept)}")
    print(f"  Dropped: {len(dropped)}")
    
    if dropped:
        print(f"\n  Dropped epochs reasons:")
        for i in dropped[:10]:
            print(f"    Epoch {i}: {epochs.drop_log[i]}")
    
    # =========================================================================
    # Test 3: Check event times vs data boundaries
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Event times vs data boundaries")
    print("=" * 80)
    
    data_duration = raw_filtered.times[-1]
    print(f"\nData duration: {data_duration:.1f}s")
    print(f"Epoch window: [{config.epochs.eeg_tmin_sec}, {config.epochs.eeg_tmax_sec}]s")
    
    print(f"\nEvent boundary check:")
    for i, event in enumerate(events):
        event_sample = event[0]
        event_time = event_sample / raw_filtered.info['sfreq']
        epoch_start = event_time + config.epochs.eeg_tmin_sec
        epoch_end = event_time + config.epochs.eeg_tmax_sec
        
        event_type = [k for k, v in event_id_found.items() if v == event[2]][0]
        
        start_ok = epoch_start >= 0
        end_ok = epoch_end <= data_duration
        
        status = "OK" if (start_ok and end_ok) else "DROPPED"
        reason = ""
        if not start_ok:
            reason += f"start<0 ({epoch_start:.1f}s) "
        if not end_ok:
            reason += f"end>data ({epoch_end:.1f}s > {data_duration:.1f}s)"
        
        print(f"  Event {i+1:2d}: {event_type:<10} t={event_time:6.1f}s -> [{epoch_start:6.1f}, {epoch_end:6.1f}]s - {status} {reason}")
    
    # =========================================================================
    # Test 4: Manual annotation embedding (bypass embed_events)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Manual annotation embedding")
    print("=" * 80)
    
    raw_test4 = raw_eeg.copy()
    
    # Get marker data
    markers = marker_stream["time_series"]
    marker_timestamps_lsl = marker_stream["time_stamps"]
    marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
    
    # Get EEG timing
    eeg_start_lsl = eeg_timestamps[0]
    eeg_end_lsl = eeg_timestamps[-1]
    
    # Manually create annotations
    onsets = []
    descriptions = []
    
    for marker, ts in zip(marker_list, marker_timestamps_lsl):
        # Skip if outside EEG range
        if ts < eeg_start_lsl or ts > eeg_end_lsl:
            continue
        
        onset = ts - eeg_start_lsl
        onsets.append(onset)
        descriptions.append(marker)
    
    print(f"\nManual annotations created: {len(onsets)}")
    
    annotations = mne.Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=descriptions,
    )
    raw_test4.set_annotations(annotations)
    
    # Count
    ann_counts = {}
    for desc in descriptions:
        ann_counts[desc] = ann_counts.get(desc, 0) + 1
    
    print(f"Annotation counts:")
    for desc, count in sorted(ann_counts.items()):
        print(f"  '{desc}': {count}")
    
    # Create epochs with manual annotations
    raw_test4_filtered = raw_test4.copy().filter(l_freq=1, h_freq=40, verbose=False)
    
    event_id_manual = {'LEFT': 1, 'RIGHT': 2, 'NOTHING': 3}
    events_manual, event_id_found_manual = mne.events_from_annotations(
        raw_test4_filtered, event_id=event_id_manual, verbose=False
    )
    
    print(f"\nEvents from manual annotations: {len(events_manual)}")
    for name, eid in event_id_found_manual.items():
        count = np.sum(events_manual[:, 2] == eid)
        print(f"  {name}: {count}")
    
    # Create epochs
    epochs_manual = mne.Epochs(
        raw_test4_filtered,
        events_manual,
        event_id=event_id_found_manual,
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=None,
        preload=True,
        verbose=False,
        reject=None,
    )
    
    print(f"\nEpochs from manual annotations: {len(epochs_manual)}")
    for cond in epochs_manual.event_id.keys():
        print(f"  {cond}: {len(epochs_manual[cond])}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
