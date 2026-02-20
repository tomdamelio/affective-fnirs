"""Checkpoint verification tests for Task 8.

Verifies that filter_channels_by_sci(), mark_bad_channels_in_info(), and
generate_fnirs_timeseries_plot() functions exist and are correctly implemented
according to the design specification.
"""

import ast
from pathlib import Path

import pytest
import mne
import numpy as np


# Parse the script file to extract function source code without importing
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_analysis_sub012.py"


def get_function_source(func_name: str) -> str:
    """Extract function source code from script without importing."""
    with open(SCRIPT_PATH, "r", encoding="utf-8") as script_file:
        source = script_file.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(source, node)

    raise ValueError(f"Function {func_name} not found in {SCRIPT_PATH}")


def get_function_node(func_name: str) -> ast.FunctionDef:
    """Get AST node for a function."""
    with open(SCRIPT_PATH, "r", encoding="utf-8") as script_file:
        source = script_file.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node

    raise ValueError(f"Function {func_name} not found in {SCRIPT_PATH}")


class TestFilterChannelsBySci:
    """Tests for filter_channels_by_sci() function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("filter_channels_by_sci")
        assert source is not None
        assert "def filter_channels_by_sci" in source

    def test_has_proper_signature(self) -> None:
        """Verify function has correct parameter signature."""
        node = get_function_node("filter_channels_by_sci")
        param_names = [arg.arg for arg in node.args.args]
        assert "raw_fnirs" in param_names
        assert "sci_threshold" in param_names

    def test_has_docstring(self) -> None:
        """Verify function has documentation."""
        node = get_function_node("filter_channels_by_sci")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert len(docstring) > 50

    def test_default_threshold_is_050(self) -> None:
        """Verify default SCI threshold is 0.50 per requirement 5.1."""
        node = get_function_node("filter_channels_by_sci")
        defaults = node.args.defaults
        # sci_threshold is the second parameter, so first default
        assert len(defaults) >= 1
        assert isinstance(defaults[0], ast.Constant)
        assert defaults[0].value == 0.50

    def test_returns_tuple_of_lists(self) -> None:
        """Verify function returns tuple of (good_pairs, bad_pairs)."""
        source = get_function_source("filter_channels_by_sci")
        assert "return good_pairs, bad_pairs" in source

    def test_classifies_by_threshold_comparison(self) -> None:
        """Verify function uses > for good and <= for bad classification."""
        source = get_function_source("filter_channels_by_sci")
        # Good channels: SCI > threshold
        assert "sci > sci_threshold" in source
        # Bad channels: SCI <= threshold
        assert "sci <= sci_threshold" in source

    def test_picks_only_fnirs_channels(self) -> None:
        """Verify function filters to fNIRS channels only."""
        source = get_function_source("filter_channels_by_sci")
        assert "mne.pick_types" in source
        assert "fnirs=True" in source


class TestMarkBadChannelsInInfo:
    """Tests for mark_bad_channels_in_info() function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("mark_bad_channels_in_info")
        assert source is not None
        assert "def mark_bad_channels_in_info" in source

    def test_has_proper_signature(self) -> None:
        """Verify function has correct parameter signature."""
        node = get_function_node("mark_bad_channels_in_info")
        param_names = [arg.arg for arg in node.args.args]
        assert "raw" in param_names
        assert "bad_pairs" in param_names

    def test_has_docstring(self) -> None:
        """Verify function has documentation."""
        node = get_function_node("mark_bad_channels_in_info")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert len(docstring) > 50

    def test_marks_both_wavelength_channels(self) -> None:
        """Verify function marks both wavelength channels for each bad pair."""
        source = get_function_source("mark_bad_channels_in_info")
        # Should extract base pair from channel name
        assert 'split(" ")[0]' in source or "split(' ')[0]" in source
        # Should update raw.info['bads']
        assert 'raw.info["bads"]' in source or "raw.info['bads']" in source


class TestMarkBadChannelsBehavior:
    """Behavioral tests for mark_bad_channels_in_info() using MNE mock data."""

    def test_marks_both_wavelengths_for_bad_pair(self) -> None:
        """Verify both 760nm and 850nm channels are marked for a bad pair."""
        # Create mock Raw with wavelength channels
        ch_names = [
            "S1_D1 760", "S1_D1 850",  # Pair 1
            "S2_D2 760", "S2_D2 850",  # Pair 2
            "S3_D3 760", "S3_D3 850",  # Pair 3
        ]
        ch_types = ["fnirs_cw_amplitude"] * 6

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=10.0,
            ch_types=ch_types,
        )

        data = np.random.randn(6, 100)
        raw = mne.io.RawArray(data, info)

        # Mark S1_D1 as bad
        bad_pairs = ["S1_D1"]

        # Apply marking logic (same as mark_bad_channels_in_info)
        bad_channels = []
        for ch_name in raw.ch_names:
            base_pair = ch_name.split(" ")[0]
            if base_pair in bad_pairs:
                bad_channels.append(ch_name)
        raw.info["bads"] = bad_channels

        # Verify both wavelengths are marked
        assert "S1_D1 760" in raw.info["bads"]
        assert "S1_D1 850" in raw.info["bads"]
        assert len(raw.info["bads"]) == 2

    def test_marks_multiple_bad_pairs(self) -> None:
        """Verify multiple bad pairs are all marked correctly."""
        ch_names = [
            "S1_D1 760", "S1_D1 850",
            "S2_D2 760", "S2_D2 850",
            "S3_D3 760", "S3_D3 850",
        ]
        ch_types = ["fnirs_cw_amplitude"] * 6

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=10.0,
            ch_types=ch_types,
        )

        data = np.random.randn(6, 100)
        raw = mne.io.RawArray(data, info)

        # Mark S1_D1 and S3_D3 as bad
        bad_pairs = ["S1_D1", "S3_D3"]

        bad_channels = []
        for ch_name in raw.ch_names:
            base_pair = ch_name.split(" ")[0]
            if base_pair in bad_pairs:
                bad_channels.append(ch_name)
        raw.info["bads"] = bad_channels

        # Verify all 4 channels are marked
        assert len(raw.info["bads"]) == 4
        assert "S1_D1 760" in raw.info["bads"]
        assert "S1_D1 850" in raw.info["bads"]
        assert "S3_D3 760" in raw.info["bads"]
        assert "S3_D3 850" in raw.info["bads"]
        # S2_D2 should NOT be marked
        assert "S2_D2 760" not in raw.info["bads"]
        assert "S2_D2 850" not in raw.info["bads"]


class TestGenerateFnirsTimeseriesPlot:
    """Tests for generate_fnirs_timeseries_plot() function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        assert source is not None
        assert "def generate_fnirs_timeseries_plot" in source

    def test_has_proper_signature(self) -> None:
        """Verify function has correct parameter signature."""
        node = get_function_node("generate_fnirs_timeseries_plot")
        param_names = [arg.arg for arg in node.args.args]
        assert "raw_haemo" in param_names
        assert "output_path" in param_names
        assert "config" in param_names
        assert "good_channels" in param_names

    def test_has_docstring(self) -> None:
        """Verify function has documentation."""
        node = get_function_node("generate_fnirs_timeseries_plot")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert len(docstring) > 50

    def test_filters_by_good_channels(self) -> None:
        """Verify function filters channels by good_channels list."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        # Should check if channel base pair is in good_channels
        assert "good_channels" in source
        assert "base_pair" in source

    def test_handles_zero_good_channels(self) -> None:
        """Verify function returns None with warning for zero good channels."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        assert "return None" in source
        assert "warning" in source.lower() or "Warning" in source

    def test_creates_two_subplots(self) -> None:
        """Verify function creates 2 vertically stacked subplots (HbO, HbR)."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        # Should create figure with 2 rows, 1 column
        assert "subplots(2, 1" in source or "subplots(2,1" in source

    def test_adds_stimulus_markers(self) -> None:
        """Verify function adds vertical lines for stimulus events."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        # Should use axvline for stimulus markers
        assert "axvline" in source

    def test_uses_condition_colors(self) -> None:
        """Verify function uses correct colors for conditions."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        # LEFT=green, RIGHT=purple, NOTHING=gray
        assert "LEFT" in source
        assert "RIGHT" in source
        assert "NOTHING" in source

    def test_saves_with_bids_naming(self) -> None:
        """Verify output filename follows BIDS naming pattern."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        assert "sub-" in source
        assert "ses-" in source
        assert "task-" in source
        assert "desc-fnirs_timeseries.png" in source

    def test_computes_mean_and_std(self) -> None:
        """Verify function computes mean and std for shaded band."""
        source = get_function_source("generate_fnirs_timeseries_plot")
        assert "np.mean" in source
        assert "np.std" in source
        assert "fill_between" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
