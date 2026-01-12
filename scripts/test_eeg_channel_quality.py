"""
Test script for EEG channel quality assessment.

This script tests the compute_eeg_channel_quality function to ensure
it correctly evaluates channel quality based on correlation and variance metrics.
"""

import json
from pathlib import Path
import importlib

import mne
import numpy as np

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_eeg_raw

# Force reload to get latest version
import affective_fnirs.reporting
importlib.reload(affective_fnirs.reporting)
from affective_fnirs.reporting import compute_eeg_channel_quality

# Paths
xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
eeg_json = Path("data/raw/sub-002/sub-002_TomFingerTapping_eeg.json")

print("=" * 70)
print("Testing EEG Channel Quality Assessment")
print("=" * 70)

# Load XDF
print("\n1. Loading XDF file...")
streams, header = load_xdf_file(xdf_file)
print(f"   Loaded {len(streams)} streams")

# Identify streams
print("\n2. Identifying streams...")
identified_streams = identify_streams(streams)
print(f"   Found: {', '.join(identified_streams.keys()).upper()}")

# Extract EEG data
print("\n3. Extracting EEG stream...")
eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
print(f"   EEG: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels, {eeg_sfreq} Hz")

# Build EEG Raw
print("\n4. Building EEG Raw object...")
raw_eeg = build_eeg_raw(
    eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
)
print(f"   EEG Raw: {len(raw_eeg.ch_names)} channels, {raw_eeg.times[-1]:.1f}s duration")

# Compute channel quality
print("\n5. Computing EEG channel quality...")
channels_to_evaluate = ["C3", "C4", "F3", "F4", "Fp1", "Fp2"]

# For sub-002, only C3, C4, F3, F4 were well-connected
known_good_channels = ["C3", "C4", "F3", "F4"]

channel_quality = compute_eeg_channel_quality(
    raw_eeg, 
    channels_to_evaluate,
    known_good_channels=known_good_channels
)

print("\n" + "=" * 70)
print("EEG Channel Quality Results")
print("=" * 70)
print(f"{'Channel':<10} {'Correlation':<15} {'Variance':<15} {'Quality':<10}")
print("-" * 70)

for ch_quality in channel_quality:
    # Color coding for terminal
    if ch_quality.quality_status == "good":
        status_str = "✓ GOOD"
    elif ch_quality.quality_status == "fair":
        status_str = "⚠ FAIR"
    else:
        status_str = "✗ POOR"

    print(
        f"{ch_quality.channel_name:<10} "
        f"{ch_quality.mean_correlation:<15.3f} "
        f"{ch_quality.signal_variance:<15.2e} "
        f"{status_str:<10}"
    )

print("=" * 70)

# Summary
good_count = sum(1 for ch in channel_quality if ch.quality_status == "good")
fair_count = sum(1 for ch in channel_quality if ch.quality_status == "fair")
poor_count = sum(1 for ch in channel_quality if ch.quality_status == "poor")

print(f"\nSummary: {good_count} good, {fair_count} fair, {poor_count} poor")

if poor_count > 0:
    poor_channels = [ch.channel_name for ch in channel_quality if ch.quality_status == "poor"]
    print(f"⚠ Poor quality channels: {', '.join(poor_channels)}")
    print("  These channels may have poor electrode contact or excessive noise.")

print("\n✓ Test complete!")
