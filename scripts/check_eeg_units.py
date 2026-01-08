"""Check EEG data units and scaling."""

import numpy as np
import pyxdf

streams, _ = pyxdf.load_xdf(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)

for s in streams:
    if s["info"]["type"][0] == "EEG" and int(s["info"]["channel_count"][0]) > 10:
        eeg_stream = s
        break

data = np.array(eeg_stream["time_series"])
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min: {np.min(data):.2f}")
print(f"Max: {np.max(data):.2f}")
print(f"Mean: {np.mean(data):.2f}")
print(f"Std: {np.std(data):.2f}")

# Check units from stream info
try:
    channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    for i, ch in enumerate(channels[:3]):
        print(f"\nChannel {i}: {ch.get('label', ['?'])[0]}")
        print(f"  Unit: {ch.get('unit', ['?'])[0]}")
        print(f"  Type: {ch.get('type', ['?'])[0]}")
except Exception as e:
    print(f"Could not read channel info: {e}")

# Typical EEG values
print("\n--- Expected EEG ranges ---")
print("Typical EEG amplitude: 10-100 µV")
print("In Volts: 1e-5 to 1e-4 V")
print("In nanovolts: 10,000 to 100,000 nV")

# What scaling would give us reasonable values?
print("\n--- Scaling analysis ---")
typical_eeg_uv = 50  # µV
current_std = np.std(data)
needed_scale = typical_eeg_uv / current_std
print(f"Current std: {current_std:.2e}")
print(f"To get ~50 µV std, need to scale by: {needed_scale:.2e}")
