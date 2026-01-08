"""Verify EEG scaling is correct."""

import json
import logging
from pathlib import Path

import numpy as np
import pyxdf

from affective_fnirs.mne_builder import build_eeg_raw

logging.basicConfig(level=logging.INFO)

# Load XDF
streams, _ = pyxdf.load_xdf(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)

# Find EEG stream
eeg_stream = None
for s in streams:
    if s["info"]["type"][0] == "EEG" and int(s["info"]["channel_count"][0]) > 10:
        eeg_stream = s
        break

data = np.array(eeg_stream["time_series"])
timestamps = np.array(eeg_stream["time_stamps"])
srate = float(eeg_stream["info"]["nominal_srate"][0])

print("=== RAW DATA ===")
print(f"Std: {np.std(data):.2f}")
print(f"Max: {np.max(np.abs(data)):.2f}")

# Load channel names
with open("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json") as f:
    cfg = json.load(f)

ch_names = [ch.strip() for ch in cfg["Channels"].split(",")][:32]

# Build Raw
raw = build_eeg_raw(data, srate, ch_names, None, timestamps)

print("\n=== AFTER build_eeg_raw ===")
raw_data = raw.get_data()
print(f"Std (V): {np.std(raw_data):.2e}")
print(f"Std (µV): {np.std(raw_data)*1e6:.2f}")
print(f"Max (V): {np.max(np.abs(raw_data)):.2e}")
print(f"Max (µV): {np.max(np.abs(raw_data))*1e6:.2f}")

print("\n=== EXPECTED ===")
print("Typical EEG std: 10-100 µV")
print("Typical EEG max: 100-500 µV")
