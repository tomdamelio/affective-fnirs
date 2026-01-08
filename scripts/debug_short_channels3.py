"""Debug script to check channel types."""
import json
from pathlib import Path
from affective_fnirs.ingestion import *
from affective_fnirs.mne_builder import *

# Load data
streams, _ = load_xdf_file(Path('data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf'))
identified = identify_streams(streams)
fnirs_data, fnirs_sfreq, fnirs_ts = extract_stream_data(identified['fnirs'])

# Load JSON
fnirs_meta = json.load(open('data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json'))

# Build Raw
raw = build_fnirs_raw(fnirs_data, fnirs_sfreq, fnirs_meta['ChMontage'], fnirs_ts)

print("Channel types:")
for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()):
    if 'S13' in ch or 'S14' in ch or 'AUX' in ch:
        print(f"  {ch}: {ch_type}")
