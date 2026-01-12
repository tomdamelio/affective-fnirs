"""
Detailed diagnostic script for EEG channel quality.

Investigates why Fp1 and Fp2 show good quality when they may not be connected.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_eeg_raw

# Paths
xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")

print("=" * 80)
print("DETAILED EEG CHANNEL QUALITY DIAGNOSTIC")
print("=" * 80)

# Load data
print("\n1. Loading data...")
streams, header = load_xdf_file(xdf_file)
identified_streams = identify_streams(streams)
eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
raw_eeg = build_eeg_raw(
    eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
)
print(f"   Loaded: {len(raw_eeg.ch_names)} channels, {raw_eeg.times[-1]:.1f}s")

# Channels to investigate
channels_to_check = ["C3", "C4", "F3", "F4", "Fp1", "Fp2"]

print("\n" + "=" * 80)
print("SIGNAL STATISTICS")
print("=" * 80)

for ch_name in channels_to_check:
    if ch_name not in raw_eeg.ch_names:
        print(f"\n{ch_name}: NOT FOUND")
        continue
    
    # Get channel data
    ch_data = raw_eeg.get_data(picks=[ch_name])[0]
    
    print(f"\n{ch_name}:")
    print(f"  Mean:     {np.mean(ch_data):.6e} V")
    print(f"  Std:      {np.std(ch_data):.6e} V")
    print(f"  Min:      {np.min(ch_data):.6e} V")
    print(f"  Max:      {np.max(ch_data):.6e} V")
    print(f"  Range:    {np.ptp(ch_data):.6e} V")
    print(f"  Variance: {np.var(ch_data):.6e} V²")
    
    # Check if signal is flat (disconnected)
    unique_values = len(np.unique(ch_data))
    print(f"  Unique values: {unique_values}")
    
    # Check if signal is constant or near-constant
    if np.std(ch_data) < 1e-15:
        print(f"  ⚠ WARNING: Signal is essentially flat (std < 1e-15)")
    
    # Check for saturation
    if np.abs(np.max(ch_data)) > 1e-3 or np.abs(np.min(ch_data)) > 1e-3:
        print(f"  ⚠ WARNING: Signal may be saturated (>1mV)")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Get all EEG data
eeg_picks = mne.pick_types(raw_eeg.info, eeg=True, exclude=[])
eeg_channel_names = [raw_eeg.ch_names[i] for i in eeg_picks]
data_eeg = raw_eeg.get_data(picks=eeg_picks)

# Compute correlation matrix
corr_matrix = np.corrcoef(data_eeg)
np.fill_diagonal(corr_matrix, np.nan)

print("\nCorrelation with other channels:")
for ch_name in channels_to_check:
    if ch_name not in eeg_channel_names:
        continue
    
    ch_idx = eeg_channel_names.index(ch_name)
    ch_corr = corr_matrix[ch_idx, :]
    
    # Remove NaN (self-correlation)
    ch_corr_clean = ch_corr[~np.isnan(ch_corr)]
    
    print(f"\n{ch_name}:")
    print(f"  Mean correlation:   {np.mean(ch_corr_clean):.3f}")
    print(f"  Median correlation: {np.median(ch_corr_clean):.3f}")
    print(f"  Min correlation:    {np.min(ch_corr_clean):.3f}")
    print(f"  Max correlation:    {np.max(ch_corr_clean):.3f}")
    
    # Find channels with highest correlation
    top_corr_indices = np.argsort(ch_corr_clean)[-5:][::-1]
    print(f"  Top 5 correlated channels:")
    for idx in top_corr_indices:
        corr_ch_name = eeg_channel_names[idx if idx < ch_idx else idx + 1]
        corr_value = ch_corr_clean[idx]
        print(f"    {corr_ch_name}: {corr_value:.3f}")

print("\n" + "=" * 80)
print("VISUAL INSPECTION (First 5 seconds)")
print("=" * 80)

# Plot first 5 seconds of each channel
fig, axes = plt.subplots(len(channels_to_check), 1, figsize=(12, 10), sharex=True)
fig.suptitle("EEG Channel Signals (First 5 seconds)", fontsize=14, fontweight="bold")

for idx, ch_name in enumerate(channels_to_check):
    if ch_name not in raw_eeg.ch_names:
        axes[idx].text(0.5, 0.5, f"{ch_name}: NOT FOUND", 
                      ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_ylabel(ch_name)
        continue
    
    # Get 5 seconds of data
    ch_data = raw_eeg.get_data(picks=[ch_name], start=0, stop=int(5 * eeg_sfreq))[0]
    times = np.arange(len(ch_data)) / eeg_sfreq
    
    axes[idx].plot(times, ch_data * 1e6, linewidth=0.5)  # Convert to µV
    axes[idx].set_ylabel(f"{ch_name}\n(µV)")
    axes[idx].grid(True, alpha=0.3)
    
    # Add statistics to plot
    mean_val = np.mean(ch_data) * 1e6
    std_val = np.std(ch_data) * 1e6
    axes[idx].text(0.02, 0.95, f"μ={mean_val:.2f}, σ={std_val:.2f}", 
                  transform=axes[idx].transAxes, va='top', fontsize=8,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("outputs/eeg_channel_quality_diagnostic.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved plot to: outputs/eeg_channel_quality_diagnostic.png")

print("\n" + "=" * 80)
print("POWER SPECTRAL DENSITY")
print("=" * 80)

# Compute PSD for each channel
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Power Spectral Density (0-50 Hz)", fontsize=14, fontweight="bold")
axes = axes.flatten()

for idx, ch_name in enumerate(channels_to_check):
    if ch_name not in raw_eeg.ch_names:
        axes[idx].text(0.5, 0.5, f"{ch_name}: NOT FOUND", 
                      ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(ch_name)
        continue
    
    # Compute PSD using Welch's method
    psd = raw_eeg.compute_psd(picks=[ch_name], fmin=0, fmax=50, method='welch')
    freqs = psd.freqs
    psd_data = psd.get_data()[0]
    
    axes[idx].semilogy(freqs, psd_data)
    axes[idx].set_title(ch_name)
    axes[idx].set_xlabel("Frequency (Hz)")
    axes[idx].set_ylabel("Power (V²/Hz)")
    axes[idx].grid(True, alpha=0.3)
    
    # Highlight alpha and beta bands
    axes[idx].axvspan(8, 13, alpha=0.2, color='blue', label='Alpha')
    axes[idx].axvspan(13, 30, alpha=0.2, color='red', label='Beta')
    
    if idx == 0:
        axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/eeg_channel_psd_diagnostic.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved plot to: outputs/eeg_channel_psd_diagnostic.png")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nReview the plots and statistics above to determine:")
print("1. Are Fp1/Fp2 truly disconnected (flat signal)?")
print("2. Do they have abnormal correlation patterns?")
print("3. Is their PSD characteristic of noise vs. real EEG?")
