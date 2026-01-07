"""Debug script to check channel counts."""

import json
from pathlib import Path

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)

xdf_path = Path(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)
eeg_json_path = Path(
    "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json"
)

# Load XDF
streams, header = load_xdf_file(xdf_path)
identified = identify_streams(streams)

# Check EEG stream
eeg_stream = identified["eeg"]
print(f"EEG stream name: {eeg_stream['info']['name'][0]}")
print(f"EEG stream channel_count: {eeg_stream['info']['channel_count'][0]}")

eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(eeg_stream)
print(f"EEG data shape: {eeg_data.shape}")
print(f"EEG data channels: {eeg_data.shape[1]}")

# Check JSON
with open(eeg_json_path, "r") as f:
    eeg_json = json.load(f)

channel_str = eeg_json["Channels"]
channel_names = [ch.strip() for ch in channel_str.split(",")]
print(f"\nJSON channel names: {len(channel_names)}")
print(f"Channel names: {channel_names}")

# Check if stream has channel labels
if "desc" in eeg_stream["info"]:
    desc = eeg_stream["info"]["desc"][0]
    if "channels" in desc:
        channels_elem = desc["channels"][0]
        if "channel" in channels_elem:
            stream_channels = channels_elem["channel"]
            print(f"\nStream has {len(stream_channels)} channel labels")
            for i, ch in enumerate(stream_channels[:5]):
                if "label" in ch:
                    print(f"  Channel {i}: {ch['label'][0]}")
