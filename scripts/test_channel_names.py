"""Test channel naming extraction."""
import json
from pathlib import Path

# Load JSON
json_path = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json")
with open(json_path) as f:
    data = json.load(f)

# Test first few channels
print("Testing channel name extraction:")
print("=" * 70)

for i, ch in enumerate(data["ChMontage"][:5]):
    source = ch["source"]
    detector = ch["detector"]
    wavelength = ch["wavelength"]
    
    # Extract IDs
    source_id = source.split("_")[0]
    detector_id = detector.split("_")[0]
    
    # Create name
    ch_name = f"{source_id}_{detector_id} {wavelength}"
    
    print(f"Channel {i}:")
    print(f"  Source: {source} → {source_id}")
    print(f"  Detector: {detector} → {detector_id}")
    print(f"  Wavelength: {wavelength}")
    print(f"  Final name: {ch_name}")
    print()
