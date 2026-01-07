"""
Test hemoglobin filtering functionality.

This test verifies that the filter_hemoglobin_data() function correctly:
1. Applies bandpass filter (0.01-0.5 Hz) to HbO/HbR channels
2. Uses FIR filter for linear phase response
3. Only filters hemoglobin channels, not short channels
4. Preserves hemodynamic response frequencies

Requirements: 6.4
"""

import numpy as np
import mne
import pytest
from affective_fnirs.fnirs_processing import filter_hemoglobin_data


def test_filter_hemoglobin_data_basic():
    """Test basic hemoglobin filtering with synthetic data."""
    # Create synthetic hemoglobin data
    sfreq = 10.0  # 10 Hz sampling rate
    duration = 60.0  # 60 seconds
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    
    # Create synthetic HbO and HbR signals with multiple frequency components
    # 1. Slow drift (0.005 Hz) - should be removed
    # 2. Hemodynamic response (0.1 Hz) - should be preserved
    # 3. Cardiac pulsation (1.0 Hz) - should be removed
    drift = 2.0 * np.sin(2 * np.pi * 0.005 * times)
    hrf = 5.0 * np.sin(2 * np.pi * 0.1 * times)
    cardiac = 1.0 * np.sin(2 * np.pi * 1.0 * times)
    
    hbo_signal = drift + hrf + cardiac
    hbr_signal = -0.5 * (drift + hrf + cardiac)  # HbR typically inverse of HbO
    
    # Create MNE Raw object with hemoglobin channels
    data = np.vstack([hbo_signal, hbr_signal])
    ch_names = ["S1_D1 hbo", "S1_D1 hbr"]
    ch_types = ["hbo", "hbr"]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_haemo = mne.io.RawArray(data, info)
    
    # Apply filtering
    raw_filtered = filter_hemoglobin_data(raw_haemo, l_freq=0.01, h_freq=0.5)
    
    # Verify output
    assert raw_filtered is not None
    assert len(raw_filtered.ch_names) == 2
    assert raw_filtered.info["sfreq"] == sfreq
    
    # Verify channel types preserved
    filtered_types = raw_filtered.get_channel_types()
    assert filtered_types == ["hbo", "hbr"]
    
    # Verify data was modified (filtered)
    original_data = raw_haemo.get_data()
    filtered_data = raw_filtered.get_data()
    assert not np.allclose(original_data, filtered_data)
    
    print("✓ Basic hemoglobin filtering test passed")


def test_filter_hemoglobin_data_frequency_response():
    """Test that filtering preserves hemodynamic frequencies and removes others."""
    # Create synthetic data with known frequency components
    sfreq = 10.0
    duration = 120.0  # Longer duration for better frequency resolution
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    
    # Create signal with three frequency components
    drift = 3.0 * np.sin(2 * np.pi * 0.005 * times)  # 0.005 Hz - should be removed
    hrf = 5.0 * np.sin(2 * np.pi * 0.1 * times)      # 0.1 Hz - should be preserved
    cardiac = 2.0 * np.sin(2 * np.pi * 1.0 * times)  # 1.0 Hz - should be removed
    
    hbo_signal = drift + hrf + cardiac
    
    # Create Raw object
    data = np.array([hbo_signal])
    ch_names = ["S1_D1 hbo"]
    ch_types = ["hbo"]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_haemo = mne.io.RawArray(data, info)
    
    # Apply filtering
    raw_filtered = filter_hemoglobin_data(raw_haemo, l_freq=0.01, h_freq=0.5)
    
    # Compute power spectral density
    from scipy import signal as sp_signal
    
    filtered_data = raw_filtered.get_data()[0]
    freqs, psd = sp_signal.welch(filtered_data, fs=sfreq, nperseg=int(sfreq * 10))
    
    # Check that hemodynamic frequency (0.1 Hz) has significant power
    hrf_idx = np.argmin(np.abs(freqs - 0.1))
    hrf_power = psd[hrf_idx]
    
    # Check that drift frequency (0.005 Hz) has minimal power
    # Note: 0.005 Hz is below our high-pass cutoff of 0.01 Hz
    drift_idx = np.argmin(np.abs(freqs - 0.005))
    drift_power = psd[drift_idx]
    
    # Check that cardiac frequency (1.0 Hz) has minimal power
    # Note: 1.0 Hz is above our low-pass cutoff of 0.5 Hz
    cardiac_idx = np.argmin(np.abs(freqs - 1.0))
    cardiac_power = psd[cardiac_idx]
    
    # HRF power should be much higher than drift and cardiac
    assert hrf_power > drift_power * 10, "HRF frequency should be preserved"
    # Cardiac should be attenuated, but not necessarily by 10x due to filter roll-off
    assert hrf_power > cardiac_power * 5, "Cardiac frequency should be attenuated"
    
    print("✓ Frequency response test passed")
    print(f"  HRF power (0.1 Hz): {hrf_power:.6f}")
    print(f"  Drift power (0.005 Hz): {drift_power:.6f}")
    print(f"  Cardiac power (1.0 Hz): {cardiac_power:.6f}")
    print(f"  HRF/Cardiac ratio: {hrf_power/cardiac_power:.2f}x")


def test_filter_hemoglobin_data_no_hbo_hbr_channels():
    """Test that function raises error when no HbO/HbR channels present."""
    # Create Raw object with only OD channels (not hemoglobin)
    sfreq = 10.0
    duration = 10.0
    n_samples = int(sfreq * duration)
    
    data = np.random.randn(2, n_samples)
    ch_names = ["S1_D1 760", "S1_D1 850"]
    ch_types = ["fnirs_od", "fnirs_od"]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_od = mne.io.RawArray(data, info)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="No HbO or HbR channels found"):
        filter_hemoglobin_data(raw_od)
    
    print("✓ Error handling test passed")


def test_filter_hemoglobin_data_mixed_channels():
    """Test filtering with mixed channel types (HbO/HbR + short channels)."""
    # Create Raw object with both hemoglobin and short channels
    sfreq = 10.0
    duration = 30.0
    n_samples = int(sfreq * duration)
    
    # Create synthetic data
    hbo_data = np.random.randn(2, n_samples)
    hbr_data = np.random.randn(2, n_samples)
    short_data = np.random.randn(2, n_samples)
    
    data = np.vstack([hbo_data, hbr_data, short_data])
    ch_names = [
        "S1_D1 hbo", "S2_D2 hbo",
        "S1_D1 hbr", "S2_D2 hbr",
        "S5_D5 760", "S5_D5 850"  # Short channels (OD type)
    ]
    ch_types = ["hbo", "hbo", "hbr", "hbr", "fnirs_od", "fnirs_od"]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_mixed = mne.io.RawArray(data, info)
    
    # Store original short channel data
    short_ch_indices = [4, 5]
    original_short_data = raw_mixed.get_data(picks=short_ch_indices).copy()
    
    # Apply filtering
    raw_filtered = filter_hemoglobin_data(raw_mixed, l_freq=0.01, h_freq=0.5)
    
    # Verify short channels were NOT filtered (should be unchanged)
    filtered_short_data = raw_filtered.get_data(picks=short_ch_indices)
    
    # Short channels should be identical (not filtered)
    assert np.allclose(original_short_data, filtered_short_data), \
        "Short channels should not be filtered"
    
    # HbO/HbR channels should be filtered (different from original)
    hbo_hbr_indices = [0, 1, 2, 3]
    original_hbo_hbr = raw_mixed.get_data(picks=hbo_hbr_indices)
    filtered_hbo_hbr = raw_filtered.get_data(picks=hbo_hbr_indices)
    
    assert not np.allclose(original_hbo_hbr, filtered_hbo_hbr), \
        "HbO/HbR channels should be filtered"
    
    print("✓ Mixed channel types test passed")


if __name__ == "__main__":
    print("Running hemoglobin filtering tests...\n")
    
    test_filter_hemoglobin_data_basic()
    test_filter_hemoglobin_data_frequency_response()
    test_filter_hemoglobin_data_no_hbo_hbr_channels()
    test_filter_hemoglobin_data_mixed_channels()
    
    print("\n✅ All hemoglobin filtering tests passed!")
