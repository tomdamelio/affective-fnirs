#!/usr/bin/env python3
"""Check channel distances after montage application."""
import sys
from pathlib import Path
sys.path.insert(0, "src")

import json
import numpy as np
import mne
from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_fnirs_raw

# Load XDF
xdf_path = Path("data/raw/sub-012/ses-001/sub-12_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
stream_dict = identify_streams(streams)
fnirs_stream = stream_dict.get("fnirs")

# Load montage
montage_path = "data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json"
with open(montage_path) as f:
    montage_json = json.load(f)

# Build raw
fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
raw_fnirs = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_json["ChMontage"], fnirs_timestamps)

# Check distances before montage application
print("Distances BEFORE montage application:")
for ch_idx, ch_name in enumerate(raw_fnirs.ch_names[:10]):
    dist = raw_fnirs.info["chs"][ch_idx]["loc"][10]
    print(f"  {ch_name}: {dist*1000:.1f} mm")

# Apply montage
ch_montage = montage_json.get("ChMontage", [])
sources = {}
detectors = {}

for ch_info in ch_montage:
    source_id = ch_info["source"].split("_")[0]
    detector_id = ch_info["detector"].split("_")[0]
    source_xyz = ch_info.get("source_xyz_m")
    detector_xyz = ch_info.get("detector_xyz_m")
    
    if source_xyz is not None and source_id not in sources:
        sources[source_id] = tuple(source_xyz)
    if detector_xyz is not None and detector_id not in detectors:
        detectors[detector_id] = tuple(detector_xyz)

print(f"\nFound {len(sources)} sources, {len(detectors)} detectors")

# Apply positions
for ch_idx, ch_name in enumerate(raw_fnirs.ch_names):
    parts = ch_name.split(" ")
    if len(parts) != 2:
        continue
    
    pair = parts[0]
    pair_parts = pair.split("_")
    if len(pair_parts) != 2:
        continue
    
    source_id = pair_parts[0]
    detector_id = pair_parts[1]
    
    source_pos = sources.get(source_id)
    detector_pos = detectors.get(detector_id)
    
    if source_pos is None or detector_pos is None:
        continue
    
    raw_fnirs.info["chs"][ch_idx]["loc"][0:3] = source_pos
    raw_fnirs.info["chs"][ch_idx]["loc"][3:6] = detector_pos
    
    distance = np.sqrt(
        (source_pos[0] - detector_pos[0])**2 +
        (source_pos[1] - detector_pos[1])**2 +
        (source_pos[2] - detector_pos[2])**2
    )
    raw_fnirs.info["chs"][ch_idx]["loc"][10] = distance

# Check distances after
print("\nDistances AFTER montage application (first 10 + short channels):")
for ch_idx, ch_name in enumerate(raw_fnirs.ch_names):
    dist = raw_fnirs.info["chs"][ch_idx]["loc"][10]
    if ch_idx < 10 or "S13" in ch_name or "S14" in ch_name or "S15" in ch_name or "S16" in ch_name:
        print(f"  {ch_name}: {dist*1000:.1f} mm")
