"""Check available EEG channels in sub-002 data."""
from pathlib import Path
from affective_fnirs.ingestion import load_xdf_file, identify_streams

xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_file)
identified = identify_streams(streams)
eeg_info = identified['eeg']['info']
channels = eeg_info['desc'][0]['channels'][0]['channel']

print(f"Total EEG channels: {len(channels)}")
print("\nChannel names:")
for ch in channels:
    print(f"  {ch['label'][0]}")
