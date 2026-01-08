"""Debug EEG scaling in the pipeline."""

from pathlib import Path
import json
import numpy as np
import pyxdf
import mne

# Load XDF
streams, _ = pyxdf.load_xdf(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)

for s in streams:
    if s["info"]["type"][0] == "EEG" and int(s["info"]["channel_count"][0]) > 10:
        eeg_stream = s
        break

data = np.array(eeg_stream["time_series"])
print("=== RAW XDF DATA ===")
print(f"Shape: {data.shape}")
print(f"Max abs: {np.max(np.abs(data)):.2f}")
print(f"Std: {np.std(data):.2f}")

# Simulate mne_builder scaling logic
data_max = np.max(np.abs(data))
print(f"\ndata_max = {data_max:.2f}")

if data_max > 1e6:
    scale_factor = 1e-9
    print(f"Condition: data_max > 1e6 -> scale = 1e-9")
elif data_max > 1000:
    scale_factor = 1e-6
    print(f"Condition: data_max > 1000 -> scale = 1e-6")
elif data_max > 1:
    scale_factor = 1e-3
    print(f"Condition: data_max > 1 -> scale = 1e-3")
else:
    scale_factor = 1.0
    print(f"Condition: else -> scale = 1.0")

data_scaled = data * scale_factor
print(f"\n=== AFTER SCALING (x{scale_factor}) ===")
print(f"Max abs: {np.max(np.abs(data_scaled)):.2e}")
print(f"Std: {np.std(data_scaled):.2e}")

# What detect_bad_channels sees (converts V to µV)
data_in_uv = data_scaled * 1e6
print(f"\n=== WHAT detect_bad_channels SEES (x1e6) ===")
print(f"Max abs: {np.max(np.abs(data_in_uv)):.2f} µV")
print(f"Std: {np.std(data_in_uv):.2f} µV")

# Expected values
print(f"\n=== EXPECTED VALUES ===")
print(f"Typical EEG: 10-100 µV")
print(f"Saturation threshold: 500 µV")

# The issue: data is already in µV but we're scaling it as if it were in V
print(f"\n=== DIAGNOSIS ===")
if np.std(data_in_uv) > 1000:
    print("⚠️ Data appears over-scaled!")
    print("   The XDF data is likely already in µV")
    print("   Scaling by 1e-6 then 1e6 gives original values")
    print("   But original values are ~100,000 µV which is way too high")
    print()
    print("   Possible explanations:")
    print("   1. Data is in ADC units, not µV (despite stream metadata)")
    print("   2. There's a DC offset that needs removal")
    print("   3. The amplifier gain was set incorrectly")

# Check if removing DC offset helps
data_centered = data - np.mean(data, axis=0)
print(f"\n=== AFTER DC OFFSET REMOVAL ===")
print(f"Max abs: {np.max(np.abs(data_centered)):.2f}")
print(f"Std: {np.std(data_centered):.2f}")

# This is what we should see for typical EEG
# If std is ~50-100, data is in µV
# If std is ~50000-100000, data is in nV or ADC units
