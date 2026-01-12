"""Compare timing across all subjects to identify the sub-004 issue."""

from pathlib import Path
import pyxdf

subjects = {
    "sub-002": "data/raw/sub-002/sub-002_ses-001_task-fingertapping_recording.xdf",
    "sub-003": "data/raw/sub-003/sub-003_ses-001_task-fingertapping_recording.xdf",
    "sub-004": "data/raw/sub-004/sub-004_ses-001_task-fingertapping_recording.xdf",
}

print("=" * 80)
print("CROSS-SUBJECT TIMING COMPARISON")
print("=" * 80)

for subject, xdf_path in subjects.items():
    xdf_file = Path(xdf_path)
    if not xdf_file.exists():
        print(f"\n{subject}: FILE NOT FOUND")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"{subject.upper()}")
    print("=" * 80)
    
    streams, _ = pyxdf.load_xdf(str(xdf_file))
    
    eeg_stream = None
    fnirs_stream = None
    marker_stream = None
    
    for stream in streams:
        name = stream["info"]["name"][0]
        stype = stream["info"]["type"][0]
        
        if stype == "EEG" and "actiCHamp" in name:
            eeg_stream = stream
        elif stype == "NIRS" and "RAW" in name:
            fnirs_stream = stream
        elif stype == "Markers" and name in ["cortivision_markers", "eeg_markers"]:
            if marker_stream is None:
                marker_stream = stream
    
    if eeg_stream:
        eeg_ts = eeg_stream["time_stamps"]
        eeg_duration = eeg_ts[-1] - eeg_ts[0]
        eeg_samples = len(eeg_ts)
        print(f"EEG: {eeg_duration:.1f}s ({eeg_duration/60:.1f} min), {eeg_samples} samples")
    else:
        print("EEG: NOT FOUND")
    
    if fnirs_stream:
        fnirs_ts = fnirs_stream["time_stamps"]
        fnirs_duration = fnirs_ts[-1] - fnirs_ts[0]
        fnirs_samples = len(fnirs_ts)
        print(f"fNIRS: {fnirs_duration:.1f}s ({fnirs_duration/60:.1f} min), {fnirs_samples} samples")
    else:
        print("fNIRS: NOT FOUND")
    
    if marker_stream:
        marker_ts = marker_stream["time_stamps"]
        n_markers = len(marker_ts)
        marker_duration = marker_ts[-1] - marker_ts[0] if n_markers > 1 else 0
        print(f"Markers: {n_markers} events over {marker_duration:.1f}s")
        
        # Check coverage
        if eeg_stream and n_markers > 0:
            eeg_start = eeg_ts[0]
            eeg_end = eeg_ts[-1]
            markers_in_eeg = sum(1 for t in marker_ts if eeg_start <= t <= eeg_end)
            print(f"Markers in EEG window: {markers_in_eeg}/{n_markers} ({100*markers_in_eeg/n_markers:.0f}%)")
    else:
        print("Markers: NOT FOUND")
    
    # Duration ratio
    if eeg_stream and fnirs_stream:
        ratio = eeg_duration / fnirs_duration * 100
        print(f"\nEEG/fNIRS duration ratio: {ratio:.1f}%")
        if ratio < 50:
            print("⚠️  WARNING: EEG recording is significantly shorter than fNIRS!")
