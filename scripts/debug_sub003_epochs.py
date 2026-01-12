"""Debug fNIRS epoch creation for sub-003."""

import json
from pathlib import Path

import mne
import numpy as np

from affective_fnirs.config import PipelineConfig
from affective_fnirs.ingestion import identify_streams, extract_stream_data, load_xdf_file
from affective_fnirs.mne_builder import build_fnirs_raw, embed_events
from affective_fnirs.fnirs_processing import process_fnirs_pipeline


def main():
    """Debug epoch creation for sub-003."""
    # Load config
    config = PipelineConfig.from_yaml(Path("configs/sub003_optimized.yml"))
    
    # Load XDF
    xdf_file = Path("data/raw/sub-003/sub-003_ses-001_task-fingertapping_recording.xdf")
    streams, header = load_xdf_file(xdf_file)
    
    # Identify streams
    identified = identify_streams(streams)
    
    # Load JSON
    with open("data/raw/sub-003/sub-003_ses-001_task-fingertapping_nirs.json") as f:
        fnirs_json = json.load(f)
    
    # Extract fNIRS data
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(identified["fnirs"])
    print(f"fNIRS: {fnirs_data.shape[0]} samples, {fnirs_data.shape[1]} channels, {fnirs_sfreq:.2f} Hz")
    print(f"fNIRS duration: {fnirs_data.shape[0] / fnirs_sfreq:.2f}s")
    
    # Build fNIRS Raw
    raw_fnirs = build_fnirs_raw(
        fnirs_data, fnirs_sfreq, fnirs_json["ChMontage"], fnirs_timestamps
    )
    
    # Embed events
    raw_fnirs = embed_events(raw_fnirs, identified["markers"])
    
    # Check events
    events, event_id = mne.events_from_annotations(raw_fnirs)
    print(f"\nEvents found: {len(events)}")
    print(f"Event IDs: {event_id}")
    
    # Check event times
    print("\nEvent times (seconds from start):")
    for i, event in enumerate(events):
        event_time = event[0] / raw_fnirs.info["sfreq"]
        event_name = [k for k, v in event_id.items() if v == event[2]][0]
        print(f"  {i+1}. {event_name} at {event_time:.2f}s")
    
    # Check recording duration
    recording_duration = raw_fnirs.n_times / raw_fnirs.info["sfreq"]
    print(f"\nRecording duration: {recording_duration:.2f}s")
    
    # Process fNIRS
    print("\nProcessing fNIRS...")
    raw_fnirs_processed, _ = process_fnirs_pipeline(raw_fnirs, config)
    
    # Check epoch parameters
    tmin = config.epochs.fnirs_tmin_sec
    tmax = config.epochs.fnirs_tmax_sec
    print(f"\nEpoch parameters: tmin={tmin}s, tmax={tmax}s")
    print(f"Epoch duration: {tmax - tmin}s")
    
    # Check which events would be dropped
    print("\nChecking which epochs would be dropped:")
    for i, event in enumerate(events):
        event_time = event[0] / raw_fnirs_processed.info["sfreq"]
        event_name = [k for k, v in event_id.items() if v == event[2]][0]
        
        epoch_start = event_time + tmin
        epoch_end = event_time + tmax
        
        if epoch_start < 0:
            print(f"  {i+1}. {event_name} at {event_time:.2f}s -> DROPPED (epoch starts before recording: {epoch_start:.2f}s)")
        elif epoch_end > recording_duration:
            print(f"  {i+1}. {event_name} at {event_time:.2f}s -> DROPPED (epoch ends after recording: {epoch_end:.2f}s > {recording_duration:.2f}s)")
        else:
            print(f"  {i+1}. {event_name} at {event_time:.2f}s -> OK (epoch: {epoch_start:.2f}s to {epoch_end:.2f}s)")


if __name__ == "__main__":
    main()
