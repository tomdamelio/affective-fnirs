"""Debug script to check short channel identification."""
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

print("Raw channel names (first 10):")
for i, ch in enumerate(raw.ch_names[:10]):
    print(f"  {i}: {ch}")

print("\nShort channels from JSON:")
for ch in fnirs_meta['ChMontage']:
    if ch.get('type') == 'Short':
        source_id = ch['source'].split('_')[0]
        detector_id = ch['detector'].split('_')[0]
        wavelength = ch['wavelength']
        constructed_name = f"{source_id}_{detector_id} {wavelength}"
        print(f"  {constructed_name} (from {ch['source']}, {ch['detector']})")

print("\nLong channels from JSON (first 5):")
count = 0
for ch in fnirs_meta['ChMontage']:
    if ch.get('type') == 'Long' and count < 5:
        source_id = ch['source'].split('_')[0]
        detector_id = ch['detector'].split('_')[0]
        wavelength = ch['wavelength']
        constructed_name = f"{source_id}_{detector_id} {wavelength}"
        print(f"  {constructed_name} (from {ch['source']}, {ch['detector']})")
        count += 1
