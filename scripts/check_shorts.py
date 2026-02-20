#!/usr/bin/env python3
"""Quick check of short channel names in montage JSON."""
import json

montage_path = "data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json"
with open(montage_path) as f:
    m = json.load(f)

shorts = [c for c in m["ChMontage"] if c["type"] == "Short"]
print("Short channels in JSON:")
for c in shorts:
    source_id = c["source"].split("_")[0]
    detector_id = c["detector"].split("_")[0]
    ch_name = f"{source_id}_{detector_id} {c['wavelength']}"
    print(f"  {ch_name} (from {c['source']} -> {c['detector']})")
