"""Debug script to inspect XDF stream information."""

from pathlib import Path
import pyxdf

xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")

streams, header = pyxdf.load_xdf(str(xdf_path))

print(f"Total streams: {len(streams)}\n")

for i, stream in enumerate(streams):
    info = stream['info']
    name = info.get('name', ['Unknown'])[0]
    sfreq = info.get('nominal_srate', ['0'])[0]
    n_samples = len(stream['time_series'])
    
    print(f"Stream {i}: {name}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Samples: {n_samples}")
    print(f"  Type: {info.get('type', ['Unknown'])[0]}")
    print()
