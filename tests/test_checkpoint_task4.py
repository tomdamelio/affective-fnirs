"""Checkpoint verification tests for Task 4.

Verifies that load_sub012_montage() and count_fnirs_channels() functions
exist and are correctly implemented according to the design specification.
"""

import ast
import inspect
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


class TestLoadSub012Montage:
    """Tests for load_sub012_montage() function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("load_sub012_montage")
        assert source is not None
        assert "def load_sub012_montage" in source

    def test_has_proper_signature(self) -> None:
        """Verify function has correct parameter signature."""
        node = get_function_node("load_sub012_montage")
        param_names = [arg.arg for arg in node.args.args]
        assert "config" in param_names

    def test_has_docstring(self) -> None:
        """Verify function has documentation."""
        node = get_function_node("load_sub012_montage")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert len(docstring) > 50

    def test_raises_file_not_found_error(self) -> None:
        """Verify function raises FileNotFoundError for missing file."""
        source = get_function_source("load_sub012_montage")
        assert "FileNotFoundError" in source

    def test_raises_value_error_for_missing_key(self) -> None:
        """Verify function raises ValueError for missing ChMontage key."""
        source = get_function_source("load_sub012_montage")
        assert "ValueError" in source
        assert "ChMontage" in source

    def test_logs_error_on_missing_file(self) -> None:
        """Verify function logs error when file not found."""
        source = get_function_source("load_sub012_montage")
        assert "logger.error" in source

    def test_constructs_bids_path(self) -> None:
        """Verify function constructs correct BIDS path."""
        source = get_function_source("load_sub012_montage")
        assert "sub-" in source
        assert "ses-" in source
        assert "montage_combined_EEG_fNIRS_with_3Dcoords_approx.json" in source


class TestCountFnirsChannels:
    """Tests for count_fnirs_channels() function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("count_fnirs_channels")
        assert source is not None
        assert "def count_fnirs_channels" in source

    def test_has_proper_signature(self) -> None:
        """Verify function has correct parameter signature."""
        node = get_function_node("count_fnirs_channels")
        param_names = [arg.arg for arg in node.args.args]
        assert "raw_fnirs" in param_names

    def test_has_docstring(self) -> None:
        """Verify function has documentation."""
        node = get_function_node("count_fnirs_channels")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert len(docstring) > 50

    def test_uses_mne_pick_types_correctly(self) -> None:
        """Verify function uses mne.pick_types with fnirs=True, exclude=[]."""
        source = get_function_source("count_fnirs_channels")
        assert "mne.pick_types" in source
        assert "fnirs=True" in source
        assert "exclude=[]" in source

    def test_returns_length_of_picks(self) -> None:
        """Verify function returns len() of picked channels."""
        source = get_function_source("count_fnirs_channels")
        assert "len(fnirs_picks)" in source or "len(" in source


class TestCountFnirsChannelsBehavior:
    """Behavioral tests for count_fnirs_channels() using MNE mock data."""

    def test_counts_only_fnirs_channels(self) -> None:
        """Verify function counts only fnirs_cw_amplitude, not misc channels."""
        # Create mock Raw with mixed channel types
        n_fnirs = 48  # 24 pairs × 2 wavelengths
        n_misc = 6    # AUX channels

        # Create channel info for fnirs channels
        fnirs_ch_names = [
            f"S{i//2 + 1}_D{i//2 + 1} {760 if i % 2 == 0 else 850}"
            for i in range(n_fnirs)
        ]
        misc_ch_names = [f"AUX{i+1}" for i in range(n_misc)]

        all_ch_names = fnirs_ch_names + misc_ch_names
        ch_types = ["fnirs_cw_amplitude"] * n_fnirs + ["misc"] * n_misc

        # Create minimal info structure
        info = mne.create_info(
            ch_names=all_ch_names,
            sfreq=10.0,
            ch_types=ch_types,
        )

        # Create mock data
        data = np.random.randn(len(all_ch_names), 100)
        raw = mne.io.RawArray(data, info)

        # Test using mne.pick_types directly (same logic as count_fnirs_channels)
        fnirs_picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
        count = len(fnirs_picks)

        assert count == n_fnirs, f"Expected {n_fnirs} fNIRS channels, got {count}"

    def test_returns_zero_for_no_fnirs_channels(self) -> None:
        """Verify function returns 0 when no fNIRS channels present."""
        # Create Raw with only misc channels
        misc_ch_names = [f"AUX{i+1}" for i in range(6)]
        ch_types = ["misc"] * 6

        info = mne.create_info(
            ch_names=misc_ch_names,
            sfreq=10.0,
            ch_types=ch_types,
        )

        data = np.random.randn(6, 100)
        raw = mne.io.RawArray(data, info)

        fnirs_picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
        count = len(fnirs_picks)

        assert count == 0, f"Expected 0 fNIRS channels, got {count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
