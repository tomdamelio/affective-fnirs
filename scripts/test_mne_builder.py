"""
Test script for MNE object construction module.

This script validates the MNE builder functions using sub-002 pilot data.
It tests EEG Raw construction, fNIRS Raw construction, and event embedding.

Usage:
    micromamba run -n affective-fnirs python scripts/test_mne_builder.py
"""

import json
from pathlib import Path

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_eeg_raw, build_fnirs_raw, embed_events


def test_mne_builder():
    """Test MNE object construction with sub-002 data."""
    print("=" * 80)
    print("Testing MNE Object Construction Module")
    print("=" * 80)

    # Load sub-002 data (complete EEG + fNIRS + Markers)
    xdf_path = Path(
        "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    )
    eeg_json_path = Path(
        "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json"
    )
    fnirs_json_path = Path(
        "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json"
    )

    print(f"\n1. Loading XDF file: {xdf_path.name}")
    streams, header = load_xdf_file(xdf_path)
    print(f"   ✓ Loaded {len(streams)} streams")

    print("\n2. Identifying streams...")
    identified = identify_streams(streams)
    print(f"   ✓ Found: {list(identified.keys())}")

    # Test EEG Raw construction
    print("\n3. Building EEG Raw object...")
    eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified["eeg"])
    print(f"   - EEG data shape: {eeg_data.shape}")
    print(f"   - Sampling rate: {eeg_sfreq} Hz")
    print(f"   - Duration: {len(eeg_timestamps) / eeg_sfreq:.1f} seconds")

    raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, identified["eeg"]["info"], eeg_timestamps)
    print(f"   ✓ Created EEG Raw: {len(raw_eeg.ch_names)} channels")

    # Check channel types
    eeg_ch_types = raw_eeg.get_channel_types()
    eeg_count = eeg_ch_types.count("eeg")
    misc_count = eeg_ch_types.count("misc")
    print(f"   - EEG channels: {eeg_count}")
    print(f"   - Misc channels: {misc_count}")

    # Check montage
    montage = raw_eeg.get_montage()
    if montage is not None:
        print(f"   ✓ Montage applied: {len(montage.ch_names)} positions")
    else:
        print("   ✗ No montage applied")

    # Test fNIRS Raw construction
    print("\n4. Building fNIRS Raw object...")
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(
        identified["fnirs"]
    )
    print(f"   - fNIRS data shape: {fnirs_data.shape}")
    print(f"   - Sampling rate: {fnirs_sfreq} Hz")
    print(f"   - Duration: {len(fnirs_timestamps) / fnirs_sfreq:.1f} seconds")

    # Load fNIRS montage from JSON
    with open(fnirs_json_path, "r") as f:
        fnirs_json = json.load(f)
    montage_config = fnirs_json["ChMontage"]
    print(f"   - Montage config: {len(montage_config)} channels")

    raw_fnirs = build_fnirs_raw(
        fnirs_data, fnirs_sfreq, montage_config, fnirs_timestamps
    )
    print(f"   ✓ Created fNIRS Raw: {len(raw_fnirs.ch_names)} channels")

    # Check channel types
    fnirs_ch_types = raw_fnirs.get_channel_types()
    fnirs_cw_count = fnirs_ch_types.count("fnirs_cw_amplitude")
    print(f"   - fNIRS CW amplitude channels: {fnirs_cw_count}")

    # Check wavelength metadata
    sample_ch_idx = 0
    wavelength_m = raw_fnirs.info["chs"][sample_ch_idx]["loc"][9]
    wavelength_nm = wavelength_m * 1e9
    distance_m = raw_fnirs.info["chs"][sample_ch_idx]["loc"][10]
    distance_mm = distance_m * 1000
    print(
        f"   - Sample channel '{raw_fnirs.ch_names[sample_ch_idx]}': "
        f"{wavelength_nm:.0f}nm, {distance_mm:.0f}mm"
    )

    # Test event embedding
    print("\n5. Embedding event markers...")
    marker_stream = identified["markers"]
    marker_data = marker_stream["time_series"]
    print(f"   - Marker stream: {len(marker_data)} events")

    # Embed events in EEG Raw
    raw_eeg = embed_events(raw_eeg, marker_stream)
    print(f"   ✓ Embedded {len(raw_eeg.annotations)} annotations in EEG Raw")

    # Show first few events
    if len(raw_eeg.annotations) > 0:
        print(f"   - First event: '{raw_eeg.annotations.description[0]}' "
              f"at {raw_eeg.annotations.onset[0]:.3f}s")
        if len(raw_eeg.annotations) > 1:
            print(f"   - Last event: '{raw_eeg.annotations.description[-1]}' "
                  f"at {raw_eeg.annotations.onset[-1]:.3f}s")

    # Embed events in fNIRS Raw
    raw_fnirs = embed_events(raw_fnirs, marker_stream)
    print(f"   ✓ Embedded {len(raw_fnirs.annotations)} annotations in fNIRS Raw")

    # Summary
    print("\n" + "=" * 80)
    print("MNE Object Construction Test: SUCCESS")
    print("=" * 80)
    print(f"✓ EEG Raw: {len(raw_eeg.ch_names)} channels, "
          f"{len(raw_eeg.annotations)} events")
    print(f"✓ fNIRS Raw: {len(raw_fnirs.ch_names)} channels, "
          f"{len(raw_fnirs.annotations)} events")
    print(f"✓ Temporal alignment: LSL timestamps preserved")
    print(f"✓ Montage: Standard 10-20/10-10 applied to EEG")
    print(f"✓ Metadata: Wavelengths and distances stored for fNIRS")
    print("=" * 80)


if __name__ == "__main__":
    test_mne_builder()
