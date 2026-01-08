#!/usr/bin/env python
"""Check what streams are available in sub-001 XDF file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file


def main():
    xdf_path = Path("data/raw/sub-001/sub-001_tomi_ses-001_task-fingertapping_recording.xdf")
    
    print(f"Loading: {xdf_path}")
    streams, header = load_xdf_file(xdf_path)
    
    print(f"\nFound {len(streams)} streams:")
    for i, stream in enumerate(streams):
        info = stream['info']
        name = info.get('name', ['Unknown'])[0]
        stream_type = info.get('type', ['Unknown'])[0]
        n_channels = int(info.get('channel_count', [0])[0])
        srate = info.get('nominal_srate', ['0'])[0]
        
        print(f"\nStream {i+1}:")
        print(f"  Name: {name}")
        print(f"  Type: {stream_type}")
        print(f"  Channels: {n_channels}")
        print(f"  Sample rate: {srate} Hz")
        print(f"  Samples: {len(stream['time_stamps'])}")


if __name__ == "__main__":
    main()
