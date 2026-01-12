"""Diagnose fNIRS data for sub-003."""

import json
from pathlib import Path

import mne
import pyxdf

from affective_fnirs.mne_builder import build_fnirs_raw


def main():
    """Diagnose fNIRS data."""
    # Load XDF
    xdf_file = Path("data/raw/sub-003/sub-003_ses-001_task-fingertapping_recording.xdf")
    streams, header = pyxdf.load_xdf(str(xdf_file))

    # Find fNIRS streams
    fnirs_streams = []
    for stream in streams:
        stream_type = stream["info"]["type"][0]
        stream_name = stream["info"]["name"][0]
        n_channels = stream["time_series"].shape[1] if len(stream["time_series"].shape) > 1 else 1
        n_samples = stream["time_series"].shape[0]
        print(f"Stream: {stream_name}, Type: {stream_type}, Channels: {n_channels}, Samples: {n_samples}")
        if "NIRS" in stream_type.upper():
            fnirs_streams.append(stream)

    print(f"\nFound {len(fnirs_streams)} NIRS streams")
    
    # Load JSON
    json_path = Path("data/raw/sub-003/sub-003_ses-001_task-fingertapping_nirs.json")
    with open(json_path) as json_file:
        fnirs_json = json.load(json_file)

    print(f"Channels in JSON: {len(fnirs_json['ChMontage'])}")
    
    # Find the stream that matches JSON channel count
    for stream in fnirs_streams:
        n_channels = stream["time_series"].shape[1]
        stream_name = stream["info"]["name"][0]
        print(f"\n{stream_name}: {n_channels} channels")
        if n_channels == len(fnirs_json["ChMontage"]):
            print(f"  -> MATCHES JSON config!")
        elif n_channels == len(fnirs_json["ChMontage"]) + 6:
            print(f"  -> Has 6 extra channels (likely AUX)")


if __name__ == "__main__":
    main()
