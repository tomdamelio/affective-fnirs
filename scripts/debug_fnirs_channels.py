"""Debug script to check fNIRS channel counts."""

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
fnirs_json_path = Path(
    "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json"
)

# Load XDF
streams, header = load_xdf_file(xdf_path)
identified = identify_streams(streams)

# Check fNIRS stream
fnirs_stream = identified["fnirs"]
print(f"fNIRS stream name: {fnirs_stream['info']['name'][0]}")
print(f"fNIRS stream channel_count: {fnirs_stream['info']['channel_count'][0]}")

fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
print(f"fNIRS data shape: {fnirs_data.shape}")
print(f"fNIRS data channels: {fnirs_data.shape[1]}")

# Check JSON
with open(fnirs_json_path, "r") as f:
    fnirs_json = json.load(f)

montage_config = fnirs_json["ChMontage"]
print(f"\nJSON montage config: {len(montage_config)} channels")

# Check if stream has channel labels
if "desc" in fnirs_stream["info"]:
    desc = fnirs_stream["info"]["desc"][0]
    if "channels" in desc:
        channels_elem = desc["channels"][0]
        if "channel" in channels_elem:
            stream_channels = channels_elem["channel"]
            print(f"\nStream has {len(stream_channels)} channel labels")
            for i, ch in enumerate(stream_channels):
                if "label" in ch:
                    label = ch["label"][0]
                    # Find matching JSON entry
                    json_match = None
                    for jch in montage_config:
                        if jch["channel_idx"] == i:
                            json_match = jch
                            break
                    if json_match:
                        print(f"  Channel {i}: {label} -> JSON: {json_match['location_label']}")
                    else:
                        print(f"  Channel {i}: {label} -> JSON: NO MATCH")
