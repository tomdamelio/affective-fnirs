"""Debug script to check EEG data units."""
import numpy as np
from pathlib import Path
from affective_fnirs.ingestion import *

# Load data
streams, _ = load_xdf_file(Path('data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf'))
identified = identify_streams(streams)
eeg_data, eeg_sfreq, eeg_ts = extract_stream_data(identified['eeg'])

print(f"EEG data shape: {eeg_data.shape}")
print(f"EEG data dtype: {eeg_data.dtype}")
print(f"EEG data range: min={np.min(eeg_data):.6f}, max={np.max(eeg_data):.6f}")
print(f"EEG data mean: {np.mean(eeg_data):.6f}")
print(f"EEG data std: {np.std(eeg_data):.6f}")

# Check first channel
print(f"\nFirst channel (Fp1) stats:")
print(f"  min={np.min(eeg_data[:, 0]):.6f}")
print(f"  max={np.max(eeg_data[:, 0]):.6f}")
print(f"  mean={np.mean(eeg_data[:, 0]):.6f}")
print(f"  std={np.std(eeg_data[:, 0]):.6f}")

# If data is in volts, typical EEG should be ~10-100 microvolts = 1e-5 to 1e-4 V
# If data is in microvolts, typical EEG should be ~10-100
print(f"\nIf data is in Volts, expected range: 1e-5 to 1e-4")
print(f"If data is in microvolts, expected range: 10 to 100")
