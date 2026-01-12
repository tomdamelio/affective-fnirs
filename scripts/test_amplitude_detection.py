"""
Test amplitude-based quality detection.
"""

from pathlib import Path
import numpy as np
import mne

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_eeg_raw

# Load data
xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_file)
identified_streams = identify_streams(streams)
eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps)

print("=" * 80)
print("AMPLITUDE-BASED QUALITY DETECTION TEST")
print("=" * 80)

channels_to_check = ["C3", "C4", "F3", "F4", "Fp1", "Fp2"]

for ch_name in channels_to_check:
    ch_data = raw_eeg.get_data(picks=[ch_name])[0]
    
    # Compute metrics
    std_dev = np.std(ch_data)
    amplitude_range = np.ptp(ch_data)
    mean_val = np.mean(ch_data)
    
    # Convert to µV
    std_dev_uv = std_dev * 1e6
    amplitude_range_uv = amplitude_range * 1e6
    mean_val_uv = mean_val * 1e6
    
    print(f"\n{ch_name}:")
    print(f"  Mean:      {mean_val_uv:8.2f} µV")
    print(f"  Std:       {std_dev_uv:8.2f} µV")
    print(f"  Range:     {amplitude_range_uv:8.2f} µV")
    
    # Apply classification logic
    if amplitude_range_uv < 5.0:
        status = "POOR (range < 5 µV)"
    elif amplitude_range_uv < 10.0:
        status = "FAIR (range < 10 µV)"
    elif std_dev_uv < 1.0:
        status = "POOR (std < 1 µV)"
    else:
        status = "GOOD (normal amplitude)"
    
    print(f"  Status:    {status}")

print("\n" + "=" * 80)
