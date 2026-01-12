"""Debug script for sub-004 data loading."""

from pathlib import Path
import pyxdf

xdf_file = Path("data/raw/sub-004/sub-004_ses-001_task-fingertapping_recording.xdf")
print(f"XDF exists: {xdf_file.exists()}")

print("Loading XDF...")
streams, header = pyxdf.load_xdf(str(xdf_file))
print(f"Found {len(streams)} streams:")
for s in streams:
    name = s["info"]["name"][0]
    stype = s["info"]["type"][0]
    n_samples = len(s["time_stamps"]) if "time_stamps" in s else 0
    n_channels = int(s["info"]["channel_count"][0]) if "channel_count" in s["info"] else 0
    print(f"  - {name} ({stype}): {n_samples} samples, {n_channels} channels")

# Check for markers
for s in streams:
    stype = s["info"]["type"][0]
    if stype == "Markers":
        print(f"\nMarkers found in stream: {s['info']['name'][0]}")
        if "time_series" in s:
            markers = s["time_series"]
            print(f"  Total markers: {len(markers)}")
            for i, m in enumerate(markers[:20]):
                print(f"    {i}: {m}")
