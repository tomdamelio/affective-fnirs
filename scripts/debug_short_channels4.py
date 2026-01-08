"""Debug script to check if source-detector distances are stored."""
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

print("Checking source-detector distances in channel metadata:")
for ch_name in ['S13_D3 760', 'S13_D3 850', 'S14_D4 760', 'S14_D4 850', 'S1_D1 760']:
    if ch_name in raw.ch_names:
        ch_idx = raw.ch_names.index(ch_name)
        loc = raw.info['chs'][ch_idx]['loc']
        # Distance should be in loc[10] according to build_fnirs_raw
        distance_m = loc[10] if len(loc) > 10 else None
        print(f"  {ch_name}: distance = {distance_m} m ({distance_m*1000 if distance_m else 'N/A'} mm)")
