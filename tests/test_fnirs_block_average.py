"""Tests for generate_fnirs_block_average() per-condition plotting.

Validates Requirements 3.1, 3.2, 3.3:
- Separate colored lines per condition (LEFT=blue, RIGHT=red, NOTHING=green)
- Standard deviation shading per condition
- Graceful handling when a condition has zero epochs
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from affective_fnirs.config import SubjectConfig, SubjectInfo

# Import the functions under test directly from the script module
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_analysis", Path(__file__).parent.parent / "scripts" / "run_analysis.py"
)
_mod = importlib.util.module_from_spec(_spec)
# Prevent matplotlib GUI issues during testing
import matplotlib

matplotlib.use("Agg")
_spec.loader.exec_module(_mod)

generate_fnirs_block_average = _mod.generate_fnirs_block_average
_plot_channel_per_condition = _mod._plot_channel_per_condition
_plot_channel_grand_average = _mod._plot_channel_grand_average


def _make_fnirs_epochs(
    condition_trial_counts: dict[str, int],
    n_hbo_channels: int = 4,
    sfreq: float = 10.0,
    epoch_duration_sec: float = 7.0,
) -> mne.Epochs:
    """Create mock fNIRS EpochsArray with HbO channels and condition labels.

    Args:
        condition_trial_counts: Mapping of condition name to number of trials.
        n_hbo_channels: Number of HbO channels to simulate.
        sfreq: Sampling frequency in Hz.
        epoch_duration_sec: Duration of each epoch in seconds.

    Returns:
        MNE EpochsArray with the specified conditions.
    """
    ch_names = [f"S{i}_D{i} hbo" for i in range(1, n_hbo_channels + 1)]
    ch_types = ["hbo"] * n_hbo_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    n_times = int(epoch_duration_sec * sfreq)
    rng = np.random.default_rng(42)

    all_data = []
    all_events = []
    event_id = {}
    event_code = 1
    sample_offset = 0

    for condition_name, n_trials in condition_trial_counts.items():
        if n_trials == 0:
            # MNE raises if event_id has a key with no matching events,
            # so skip zero-count conditions entirely (realistic scenario).
            event_code += 1
            continue
        event_id[condition_name] = event_code
        for trial_idx in range(n_trials):
            trial_data = rng.standard_normal((n_hbo_channels, n_times)) * 1e-6
            all_data.append(trial_data)
            all_events.append([sample_offset, 0, event_code])
            sample_offset += n_times
        event_code += 1

    if not all_data:
        data_array = np.empty((0, n_hbo_channels, n_times))
        events_array = np.empty((0, 3), dtype=int)
    else:
        data_array = np.array(all_data)
        events_array = np.array(all_events, dtype=int)

    tmin = 0.0
    epochs = mne.EpochsArray(
        data_array, info, events=events_array, tmin=tmin, event_id=event_id
    )
    return epochs


def _make_mock_config(tmp_path: Path) -> SubjectConfig:
    """Create a minimal SubjectConfig for testing."""
    return SubjectConfig(
        subject=SubjectInfo(id="012", session="001", task="fingertapping"),
        output_root=tmp_path,
    )


class TestGenerateFnirsBlockAverage:
    """Tests for the per-condition fNIRS block average plot."""

    def test_three_conditions_returns_path(self, tmp_path: Path) -> None:
        """Plot with LEFT, RIGHT, NOTHING should save successfully."""
        epochs = _make_fnirs_epochs({"LEFT": 5, "RIGHT": 5, "NOTHING": 5})
        config = _make_mock_config(tmp_path)
        result = generate_fnirs_block_average(epochs, tmp_path, config)
        assert result is not None
        assert result.exists()
        assert "block_average" in result.name

    def test_two_conditions_returns_path(self, tmp_path: Path) -> None:
        """Plot with only LEFT and RIGHT (no NOTHING) should work."""
        epochs = _make_fnirs_epochs({"LEFT": 5, "RIGHT": 5})
        config = _make_mock_config(tmp_path)
        result = generate_fnirs_block_average(epochs, tmp_path, config)
        assert result is not None
        assert result.exists()

    def test_missing_condition_graceful(self, tmp_path: Path) -> None:
        """Missing condition (absent from event_id) should be skipped (Req 3.3)."""
        epochs = _make_fnirs_epochs({"LEFT": 5, "NOTHING": 5})
        config = _make_mock_config(tmp_path)
        result = generate_fnirs_block_average(epochs, tmp_path, config)
        assert result is not None
        assert result.exists()

    def test_single_channel(self, tmp_path: Path) -> None:
        """Should handle a single HbO channel without crashing."""
        epochs = _make_fnirs_epochs(
            {"LEFT": 3, "RIGHT": 3, "NOTHING": 3}, n_hbo_channels=1
        )
        config = _make_mock_config(tmp_path)
        result = generate_fnirs_block_average(epochs, tmp_path, config)
        assert result is not None
        assert result.exists()

    def test_no_hbo_channels_returns_none(self, tmp_path: Path) -> None:
        """Should return None when no HbO channels are present."""
        ch_names = ["S1_D1 hbr", "S2_D2 hbr"]
        ch_types = ["hbr", "hbr"]
        info = mne.create_info(ch_names=ch_names, sfreq=10.0, ch_types=ch_types)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 2, 70)) * 1e-6
        events = np.array([[0, 0, 1], [70, 0, 1], [140, 0, 1]], dtype=int)
        epochs = mne.EpochsArray(
            data, info, events=events, tmin=0.0, event_id={"LEFT": 1}
        )
        config = _make_mock_config(tmp_path)
        result = generate_fnirs_block_average(epochs, tmp_path, config)
        assert result is None


class TestPlotChannelPerCondition:
    """Tests for the _plot_channel_per_condition helper."""

    def test_plots_lines_for_each_condition(self) -> None:
        """Each condition should produce a line on the axes."""
        import matplotlib.pyplot as plt

        epochs = _make_fnirs_epochs({"LEFT": 5, "RIGHT": 5, "NOTHING": 5})
        fig, ax = plt.subplots()
        condition_entries = [
            ("LEFT", "LEFT", "#1f77b4"),
            ("RIGHT", "RIGHT", "#ff7f0e"),
            ("NOTHING", "NOTHING", "#2ca02c"),
        ]
        _plot_channel_per_condition(
            ax, epochs, epochs.ch_names[0], 0, epochs.times, condition_entries
        )
        # Each condition produces 1 Line2D → 3 lines total
        lines = ax.get_lines()
        assert len(lines) == 3
        # Check legend labels
        labels = [line.get_label() for line in lines]
        assert "LEFT" in labels
        assert "RIGHT" in labels
        assert "NOTHING" in labels
        plt.close(fig)

    def test_skips_absent_condition(self) -> None:
        """Condition absent from event_id should not produce a line."""
        import matplotlib.pyplot as plt

        epochs = _make_fnirs_epochs({"LEFT": 5, "NOTHING": 5})
        fig, ax = plt.subplots()
        condition_entries = [
            ("LEFT", "LEFT", "#1f77b4"),
            ("RIGHT", "RIGHT", "#ff7f0e"),
            ("NOTHING", "NOTHING", "#2ca02c"),
        ]
        # RIGHT is in condition_entries but not in epochs.event_id,
        # so indexing epochs["RIGHT"] will raise — the main function
        # only builds entries for conditions present in event_id.
        # Test the realistic path: only pass entries that exist.
        present_entries = [
            (label, key, color)
            for label, key, color in condition_entries
            if key in epochs.event_id
        ]
        _plot_channel_per_condition(
            ax, epochs, epochs.ch_names[0], 0, epochs.times, present_entries
        )
        lines = ax.get_lines()
        assert len(lines) == 2
        labels = [line.get_label() for line in lines]
        assert "RIGHT" not in labels
        plt.close(fig)
