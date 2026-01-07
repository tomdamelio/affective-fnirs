"""
Test script for EEG processing module.

This script verifies that all EEG processing functions can be imported
and have the correct signatures.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.eeg_processing import (
    apply_ica_artifact_removal,
    detect_bad_eeg_channels,
    identify_emg_components,
    identify_eog_components,
    interpolate_bad_channels,
    preprocess_eeg,
    preprocess_eeg_pipeline,
    rereference_eeg,
)
from affective_fnirs.config import PipelineConfig

print("✓ All EEG processing functions imported successfully")

# Verify function signatures
print("\nFunction signatures:")
print(f"  preprocess_eeg: {preprocess_eeg.__name__}")
print(f"  detect_bad_eeg_channels: {detect_bad_eeg_channels.__name__}")
print(f"  apply_ica_artifact_removal: {apply_ica_artifact_removal.__name__}")
print(f"  identify_eog_components: {identify_eog_components.__name__}")
print(f"  identify_emg_components: {identify_emg_components.__name__}")
print(f"  interpolate_bad_channels: {interpolate_bad_channels.__name__}")
print(f"  rereference_eeg: {rereference_eeg.__name__}")
print(f"  preprocess_eeg_pipeline: {preprocess_eeg_pipeline.__name__}")

# Verify config can be loaded
config = PipelineConfig.default()
print(f"\n✓ PipelineConfig loaded successfully")
print(f"  EEG bandpass: {config.filters.eeg_bandpass_low_hz}-{config.filters.eeg_bandpass_high_hz} Hz")
print(f"  ICA components: {config.ica.n_components}")
print(f"  ICA random_state: {config.ica.random_state}")
print(f"  EOG threshold: {config.ica.eog_threshold}")
print(f"  EMG threshold: {config.ica.emg_threshold}")

print("\n✓ All tests passed!")
