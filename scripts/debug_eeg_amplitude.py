#!/usr/bin/env python3
"""Debug script to check EEG amplitude values."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mne
import numpy as np
from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw, embed_events

# Load config
config = SubjectConfig.from_yaml(Path("configs/sub-009.yml"))

# Load XDF
xdf_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
identified = identify_streams(streams)

# Build EEG Raw
eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified["eeg"])
event_mapping = {"LEFT": 1, "RIGHT": 2, "NOTHING": 3}
raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, identified["eeg"]["info"], eeg_timestamps)
raw_eeg = embed_events(raw_eeg, identified["markers"], event_mapping)

# Get EEG data
eeg_picks = mne.pick_types(raw_eeg.info, eeg=True, exclude=[])
data_eeg = raw_eeg.get_data(picks=eeg_picks)

# Compute amplitude metrics
print("\n" + "="*70)
print("EEG AMPLITUDE ANALYSIS (RAW DATA)")
print("="*70)
print(f"Data shape: {data_eeg.shape}")
print(f"Data dtype: {data_eeg.dtype}")
print(f"\nData statistics (in Volts):")
print(f"  Mean: {np.mean(data_eeg):.2e} V")
print(f"  Std:  {np.std(data_eeg):.2e} V")
print(f"  Min:  {np.min(data_eeg):.2e} V")
print(f"  Max:  {np.max(data_eeg):.2e} V")
print(f"  Range: {np.ptp(data_eeg):.2e} V")

print(f"\nData statistics (in microvolts):")
print(f"  Mean: {np.mean(data_eeg)*1e6:.2f} µV")
print(f"  Std:  {np.std(data_eeg)*1e6:.2f} µV")
print(f"  Min:  {np.min(data_eeg)*1e6:.2f} µV")
print(f"  Max:  {np.max(data_eeg)*1e6:.2f} µV")
print(f"  Range: {np.ptp(data_eeg)*1e6:.2f} µV")

print(f"\nPer-channel amplitude range (µV):")
for i, ch_name in enumerate([raw_eeg.ch_names[j] for j in eeg_picks]):
    amplitude_range_v = np.ptp(data_eeg[i, :])
    amplitude_range_uv = amplitude_range_v * 1e6
    std_uv = np.std(data_eeg[i, :]) * 1e6
    print(f"  {ch_name:6s}: range={amplitude_range_uv:8.2f} µV, std={std_uv:6.2f} µV")

print("="*70)
