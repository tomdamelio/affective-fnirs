"""Debug script to check if short channels are in Raw object."""
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

print(f"Total channels in Raw: {len(raw.ch_names)}")
print(f"Total channels in JSON: {len(fnirs_meta['ChMontage'])}")

# Check if short channels are in Raw
short_channel_names = []
for ch in fnirs_meta['ChMontage']:
    if ch.get('type') == 'Short':
        source_id = ch['source'].split('_')[0]
        detector_id = ch['detector'].split('_')[0]
        wavelength = ch['wavelength']
        constructed_name = f"{source_id}_{detector_id} {wavelength}"
        short_channel_names.append(constructed_name)
        
        if constructed_name in raw.ch_names:
            print(f"✓ {constructed_name} found in Raw")
        else:
            print(f"✗ {constructed_name} NOT found in Raw")

print(f"\nAll Raw channel names:")
for ch in raw.ch_names:
    print(f"  {ch}")
