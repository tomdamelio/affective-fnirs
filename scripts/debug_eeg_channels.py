"""Debug script to check EEG channel types."""
import json
from pathlib import Path
from affective_fnirs.ingestion import *
from affective_fnirs.mne_builder import *

# Load data
streams, _ = load_xdf_file(Path('data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf'))
identified = identify_streams(streams)
eeg_data, eeg_sfreq, eeg_ts = extract_stream_data(identified['eeg'])

# Build Raw
raw = build_eeg_raw(eeg_data, eeg_sfreq, identified['eeg']['info'], eeg_ts)

print(f"Total channels: {len(raw.ch_names)}")
print(f"Channel types: {set(raw.get_channel_types())}")

print("\nChannel names and types:")
for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()):
    print(f"  {ch}: {ch_type}")

# Try to pick EEG channels
print("\nTrying to pick 'eeg' channels...")
try:
    eeg_picks = raw.copy().pick('eeg')
    print(f"Successfully picked {len(eeg_picks.ch_names)} EEG channels")
except Exception as e:
    print(f"Error picking EEG channels: {e}")
