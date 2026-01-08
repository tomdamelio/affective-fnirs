"""Debug script to check EEG stream info for unit information."""
import json
from pathlib import Path
from affective_fnirs.ingestion import load_xdf_file, identify_streams

# Load data
streams, header = load_xdf_file(Path('data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf'))
identified = identify_streams(streams)

eeg_stream = identified['eeg']
print("EEG Stream Info:")
print(f"  Name: {eeg_stream['info']['name']}")
print(f"  Type: {eeg_stream['info']['type']}")
print(f"  Channel count: {eeg_stream['info']['channel_count']}")
print(f"  Nominal srate: {eeg_stream['info']['nominal_srate']}")

# Check for unit info in desc
if 'desc' in eeg_stream['info']:
    desc = eeg_stream['info']['desc'][0]
    print(f"\nDescription keys: {list(desc.keys())}")
    
    if 'channels' in desc:
        channels = desc['channels'][0]['channel']
        print(f"\nFirst channel info:")
        for key, value in channels[0].items():
            print(f"  {key}: {value}")
