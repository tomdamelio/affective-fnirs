"""Check sub-002 timing with correct filename."""

from pathlib import Path
import pyxdf

xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")

print("=" * 80)
print("SUB-002 TIMING")
print("=" * 80)

streams, _ = pyxdf.load_xdf(str(xdf_file))

for stream in streams:
    name = stream["info"]["name"][0]
    stype = stream["info"]["type"][0]
    
    if "time_stamps" in stream and len(stream["time_stamps"]) > 0:
        ts = stream["time_stamps"]
        duration = ts[-1] - ts[0]
        n_samples = len(ts)
        
        if stype in ["EEG", "NIRS"]:
            print(f"{name} ({stype}): {duration:.1f}s ({duration/60:.1f} min), {n_samples} samples")
        elif stype == "Markers" and n_samples > 0:
            print(f"{name} ({stype}): {n_samples} markers over {duration:.1f}s")

# Get EEG and marker coverage
eeg_stream = None
marker_stream = None

for stream in streams:
    name = stream["info"]["name"][0]
    stype = stream["info"]["type"][0]
    
    if stype == "EEG" and "actiCHamp" in name:
        eeg_stream = stream
    elif stype == "Markers" and "cortivision_markers" == name:
        marker_stream = stream

if eeg_stream and marker_stream:
    eeg_ts = eeg_stream["time_stamps"]
    marker_ts = marker_stream["time_stamps"]
    eeg_start, eeg_end = eeg_ts[0], eeg_ts[-1]
    markers_in_eeg = sum(1 for t in marker_ts if eeg_start <= t <= eeg_end)
    print(f"\nMarkers in EEG window: {markers_in_eeg}/{len(marker_ts)}")
