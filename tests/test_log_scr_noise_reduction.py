"""Unit tests for log_scr_noise_reduction function.

Tests the SCR noise reduction logging functionality that computes PSD
in the systemic band (0.1-0.4 Hz) before and after short channel regression.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import mne

# Add paths for imports - scripts folder needs to be first
scripts_path = str(Path(__file__).parent.parent / "scripts")
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, scripts_path)
sys.path.insert(0, src_path)

# Mock the run_analysis module before importing run_analysis_sub012
sys.modules["run_analysis"] = MagicMock()

from run_analysis_sub012 import log_scr_noise_reduction


class TestLogScrNoiseReduction:
    """Tests for log_scr_noise_reduction function."""

    def _create_mock_fnirs_raw(
        self,
        channel_pairs: list[str],
        sfreq: float = 10.0,
        duration_seconds: float = 100.0,
        systemic_amplitude: float = 0.5,
        noise_amplitude: float = 0.1,
        seed: int = 42,
    ) -> mne.io.Raw:
        """Create mock fNIRS OD data with systemic noise.

        Args:
            channel_pairs: List of channel pair names (e.g., ["S1_D1", "S9_D2"]).
            sfreq: Sampling frequency in Hz.
            duration_seconds: Duration of data in seconds.
            systemic_amplitude: Amplitude of systemic noise (Mayer waves + respiration).
            noise_amplitude: Amplitude of random noise.
            seed: Random seed for reproducibility.

        Returns:
            MNE Raw object with mock fNIRS OD data.
        """
        np.random.seed(seed)
        n_samples = int(duration_seconds * sfreq)
        times = np.arange(n_samples) / sfreq

        # Create channel names with both wavelengths
        ch_names = []
        for pair in channel_pairs:
            ch_names.extend([f"{pair} 760", f"{pair} 850"])

        n_channels = len(ch_names)

        # Systemic noise: Mayer waves (~0.1 Hz) + respiration (~0.25 Hz)
        systemic_noise = (
            systemic_amplitude * np.sin(2 * np.pi * 0.1 * times)
            + 0.6 * systemic_amplitude * np.sin(2 * np.pi * 0.25 * times)
        )

        # Create data with systemic noise + random noise
        data = np.array([
            systemic_noise + noise_amplitude * np.random.randn(n_samples)
            for _ in range(n_channels)
        ])

        # Create MNE Raw object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["fnirs_od"] * n_channels,
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        return raw

    def test_returns_dict_with_roi_names(self) -> None:
        """Function returns dictionary with ROI names as keys."""
        raw_before = self._create_mock_fnirs_raw(["S1_D1", "S9_D2"])
        raw_after = self._create_mock_fnirs_raw(
            ["S1_D1", "S9_D2"], systemic_amplitude=0.1
        )

        roi_map = {"Left Anterior": ["S1_D1", "S9_D2"]}

        result = log_scr_noise_reduction(raw_before, raw_after, roi_map)

        assert isinstance(result, dict)
        assert "Left Anterior" in result

    def test_positive_reduction_when_noise_decreases(self) -> None:
        """Returns positive percent reduction when systemic noise decreases."""
        raw_before = self._create_mock_fnirs_raw(
            ["S1_D1", "S9_D2"], systemic_amplitude=0.5
        )
        raw_after = self._create_mock_fnirs_raw(
            ["S1_D1", "S9_D2"], systemic_amplitude=0.1
        )

        roi_map = {"Left Anterior": ["S1_D1", "S9_D2"]}

        result = log_scr_noise_reduction(raw_before, raw_after, roi_map)

        assert result["Left Anterior"] > 0, "Expected positive power reduction"

    def test_handles_missing_channels(self) -> None:
        """Returns 0.0 for ROIs with no matching channels."""
        raw_before = self._create_mock_fnirs_raw(["S1_D1"])
        raw_after = self._create_mock_fnirs_raw(["S1_D1"])

        # ROI references channels that don't exist
        roi_map = {"Missing ROI": ["S99_D99"]}

        result = log_scr_noise_reduction(raw_before, raw_after, roi_map)

        assert result["Missing ROI"] == 0.0

    def test_handles_multiple_rois(self) -> None:
        """Computes reduction for multiple ROIs."""
        raw_before = self._create_mock_fnirs_raw(
            ["S1_D1", "S9_D2", "S2_D3", "S10_D4"],
            systemic_amplitude=0.5,
        )
        raw_after = self._create_mock_fnirs_raw(
            ["S1_D1", "S9_D2", "S2_D3", "S10_D4"],
            systemic_amplitude=0.15,
        )

        roi_map = {
            "Left Anterior": ["S1_D1", "S9_D2"],
            "Right Anterior": ["S2_D3", "S10_D4"],
        }

        result = log_scr_noise_reduction(raw_before, raw_after, roi_map)

        assert len(result) == 2
        assert "Left Anterior" in result
        assert "Right Anterior" in result

    def test_handles_empty_roi_map(self) -> None:
        """Returns empty dict for empty ROI map."""
        raw_before = self._create_mock_fnirs_raw(["S1_D1"])
        raw_after = self._create_mock_fnirs_raw(["S1_D1"])

        roi_map: dict[str, list[str]] = {}

        result = log_scr_noise_reduction(raw_before, raw_after, roi_map)

        assert result == {}
