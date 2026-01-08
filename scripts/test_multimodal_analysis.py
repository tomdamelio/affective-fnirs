"""
Test script for multimodal analysis module.

This script verifies that the multimodal analysis functions work correctly
with synthetic data.
"""

import numpy as np
import mne
from affective_fnirs.multimodal_analysis import (
    extract_eeg_envelope,
    resample_to_fnirs,
    compute_neurovascular_coupling,
    plot_coupling_overlay,
)


def create_synthetic_eeg_raw(duration_s=30.0, sfreq=500.0):
    """Create synthetic EEG Raw object with alpha oscillations."""
    n_channels = 5
    n_samples = int(duration_s * sfreq)
    times = np.arange(n_samples) / sfreq

    # Create synthetic data with alpha oscillations (10 Hz)
    data = np.zeros((n_channels, n_samples))
    for ch_idx in range(n_channels):
        # Add alpha oscillation with task-related modulation
        alpha_freq = 10.0  # Hz
        # Simulate ERD: alpha power decreases during task (5-20s)
        alpha_amplitude = np.ones(n_samples)
        task_mask = (times >= 5.0) & (times <= 20.0)
        alpha_amplitude[task_mask] = 0.5  # 50% power reduction during task

        data[ch_idx] = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * times)
        # Add some noise
        data[ch_idx] += 0.1 * np.random.randn(n_samples)

    # Create MNE Info
    ch_names = ["C3", "C4", "Cz", "Fp1", "Fp2"]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)

    return raw


def create_synthetic_fnirs_data(duration_s=30.0, sfreq=8.12):
    """Create synthetic fNIRS HbO time series with HRF."""
    n_samples = int(duration_s * sfreq)
    times = np.arange(n_samples) / sfreq

    # Create synthetic HRF (canonical double-gamma)
    hbo = np.zeros(n_samples)

    # Task onset at t=5s, offset at t=20s
    task_onset = 5.0
    task_offset = 20.0

    # Simulate HRF: onset at ~2s after task, peak at ~6-8s, return by ~25s
    for t_idx, t in enumerate(times):
        if t >= task_onset + 2 and t <= task_offset + 10:
            # Simple HRF model: rise and fall
            t_rel = t - (task_onset + 2)
            if t_rel < 5:
                # Rising phase
                hbo[t_idx] = 0.5 * (1 - np.exp(-t_rel / 2))
            else:
                # Falling phase
                hbo[t_idx] = 0.5 * np.exp(-(t_rel - 5) / 5)

    # Add noise
    hbo += 0.05 * np.random.randn(n_samples)

    return times, hbo


def main():
    """Test multimodal analysis functions."""
    print("=" * 60)
    print("Testing Multimodal Analysis Module")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic EEG data...")
    raw_eeg = create_synthetic_eeg_raw(duration_s=30.0, sfreq=500.0)
    print(f"   EEG Raw: {raw_eeg.n_times} samples, {raw_eeg.info['sfreq']} Hz")

    print("\n2. Creating synthetic fNIRS data...")
    fnirs_times, fnirs_hbo = create_synthetic_fnirs_data(duration_s=30.0, sfreq=8.12)
    print(f"   fNIRS HbO: {len(fnirs_hbo)} samples, 8.12 Hz")

    # Test 1: Extract EEG envelope
    print("\n3. Testing extract_eeg_envelope()...")
    try:
        eeg_times, eeg_envelope = extract_eeg_envelope(
            raw_eeg, channel="C3", freq_band=(8.0, 12.0), envelope_lowpass_hz=0.5
        )
        print(f"   ✓ Envelope extracted: {len(eeg_envelope)} samples")
        print(f"   ✓ Time range: {eeg_times[0]:.2f}s to {eeg_times[-1]:.2f}s")
        print(f"   ✓ Envelope range: [{eeg_envelope.min():.2e}, {eeg_envelope.max():.2e}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Test 2: Resample to fNIRS
    print("\n4. Testing resample_to_fnirs()...")
    try:
        eeg_envelope_resampled = resample_to_fnirs(
            eeg_envelope, eeg_times, fnirs_times, fnirs_sfreq=8.12
        )
        print(f"   ✓ Resampled: {len(eeg_envelope_resampled)} samples")
        print(f"   ✓ Matches fNIRS length: {len(eeg_envelope_resampled) == len(fnirs_hbo)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Test 3: Compute neurovascular coupling
    print("\n5. Testing compute_neurovascular_coupling()...")
    try:
        coupling = compute_neurovascular_coupling(
            eeg_envelope, fnirs_hbo, eeg_times, fnirs_times, fnirs_sfreq=8.12
        )
        print(f"   ✓ Coupling computed successfully")
        print(f"   ✓ Max correlation: {coupling['max_correlation']:.3f}")
        print(f"   ✓ Lag: {coupling['lag_seconds']:.2f}s")
        print(f"   ✓ Lag negative (EEG leads): {coupling['lag_negative']}")
        print(f"   ✓ Coupling strength: {coupling['coupling_strength']:.3f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Test 4: Plot coupling overlay
    print("\n6. Testing plot_coupling_overlay()...")
    try:
        fig = plot_coupling_overlay(
            eeg_envelope,
            fnirs_hbo,
            eeg_times,
            fnirs_times,
            coupling,
            channel_eeg="C3",
            channel_fnirs="Motor ROI (synthetic)",
            task_window=(5.0, 20.0),
            output_path=None,  # Don't save for test
        )
        print(f"   ✓ Plot generated successfully")
        print(f"   ✓ Figure has {len(fig.axes)} axes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
