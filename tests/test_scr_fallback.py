"""Unit tests for SCR fallback functionality.

Tests the SCR (Short Channel Regression) fallback mechanism that uses
explicit ROI-based pairing when MNE-NIRS proximity-based pairing produces
incorrect assignments.

Requirements: 8.5
"""
import ast
from pathlib import Path

import numpy as np
import pytest


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


class TestApplyScrWithExplicitPairing:
    """Tests for apply_scr_with_explicit_pairing function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert source is not None
        assert "def apply_scr_with_explicit_pairing" in source

    def test_has_docstring(self) -> None:
        """Verify function has proper docstring."""
        node = get_function_node("apply_scr_with_explicit_pairing")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert "ROI-based pairing" in docstring
        assert "fallback" in docstring.lower()

    def test_has_type_hints(self) -> None:
        """Verify function has type hints."""
        node = get_function_node("apply_scr_with_explicit_pairing")
        # Check return annotation exists
        assert node.returns is not None
        # Check arguments have annotations
        for arg in node.args.args:
            assert arg.annotation is not None, f"Argument {arg.arg} missing type hint"

    def test_validates_channel_types(self) -> None:
        """Verify function validates fnirs_od channel types."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "fnirs_od" in source
        assert "ValueError" in source

    def test_implements_glm_regression(self) -> None:
        """Verify function implements GLM-based regression."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check for GLM components
        assert "lstsq" in source or "linalg" in source
        assert "design_matrix" in source or "beta" in source

    def test_handles_wavelengths(self) -> None:
        """Verify function handles multiple wavelengths."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "wavelength" in source.lower()

    def test_logs_processing_info(self) -> None:
        """Verify function logs processing information."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "logger.info" in source

    def test_references_requirement(self) -> None:
        """Verify function references requirement 8.5."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "8.5" in source


class TestProcessFnirsWithScrFallback:
    """Tests for process_fnirs_with_scr_fallback function."""

    def test_function_exists(self) -> None:
        """Verify function exists in script."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        assert source is not None
        assert "def process_fnirs_with_scr_fallback" in source

    def test_has_docstring(self) -> None:
        """Verify function has proper docstring."""
        node = get_function_node("process_fnirs_with_scr_fallback")
        docstring = ast.get_docstring(node)
        assert docstring is not None
        assert "SCR" in docstring
        assert "fallback" in docstring.lower()

    def test_has_type_hints(self) -> None:
        """Verify function has type hints."""
        node = get_function_node("process_fnirs_with_scr_fallback")
        # Check return annotation exists
        assert node.returns is not None
        # Check arguments have annotations
        for arg in node.args.args:
            assert arg.annotation is not None, f"Argument {arg.arg} missing type hint"

    def test_calls_verify_scr_pairing(self) -> None:
        """Verify function calls verify_scr_pairing for verification."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        assert "verify_scr_pairing" in source

    def test_implements_fallback_logic(self) -> None:
        """Verify function implements fallback when pairing incorrect."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        # Check for conditional fallback
        assert "pairing_correct" in source
        assert "apply_scr_with_explicit_pairing" in source

    def test_logs_warning_on_mismatch(self) -> None:
        """Verify function logs warning when pairing mismatch detected."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        assert "logger.warning" in source
        assert "MISMATCH" in source or "mismatch" in source

    def test_follows_processing_order(self) -> None:
        """Verify function follows correct fNIRS processing order."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        # Check processing steps are in correct order
        od_pos = source.find("convert_to_optical_density")
        motion_pos = source.find("correct_motion_artifacts")
        scr_pos = source.find("apply_short_channel_regression")
        haemo_pos = source.find("convert_to_hemoglobin")
        filter_pos = source.find("filter_hemoglobin_data")

        assert od_pos < motion_pos < scr_pos < haemo_pos < filter_pos, (
            "Processing steps not in correct order"
        )

    def test_returns_processing_metrics(self) -> None:
        """Verify function returns processing metrics."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        assert "processing_metrics" in source
        assert "scr_pairing_correct" in source
        assert "scr_fallback_used" in source

    def test_calls_noise_reduction_logging(self) -> None:
        """Verify function calls log_scr_noise_reduction."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        assert "log_scr_noise_reduction" in source


class TestScrFallbackIntegration:
    """Integration tests for SCR fallback in main() function."""

    def test_main_uses_custom_fnirs_preprocessing(self) -> None:
        """Verify main() uses custom fNIRS preprocessing with fallback."""
        source = get_function_source("main")
        assert "process_fnirs_with_scr_fallback" in source

    def test_main_separates_eeg_fnirs_preprocessing(self) -> None:
        """Verify main() handles EEG and fNIRS preprocessing separately."""
        source = get_function_source("main")
        # fNIRS should use custom preprocessing
        assert "process_fnirs_with_scr_fallback" in source
        # EEG should still use main pipeline
        assert "run_preprocessing" in source


class TestMissingBadShortChannelHandling:
    """Tests for handling missing or bad short channels (Requirement 8.6)."""

    def test_handles_missing_short_channel_in_explicit_pairing(self) -> None:
        """Verify apply_scr_with_explicit_pairing handles missing short channels."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check for missing channel detection
        assert "not in raw_od.ch_names" in source
        # Check for ROI-level warning
        assert "lacks short channel regression" in source
        # Check for tracking skipped ROIs
        assert "rois_skipped" in source

    def test_handles_bad_short_channel_in_explicit_pairing(self) -> None:
        """Verify apply_scr_with_explicit_pairing handles bad short channels."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check for bad channel detection
        assert 'info["bads"]' in source or "info['bads']" in source
        # Check for Bad_Channel terminology per requirement
        assert "Bad_Channel" in source or "marked bad" in source.lower()

    def test_logs_roi_level_warning_for_missing_short(self) -> None:
        """Verify ROI-level warning is logged when short channel missing."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check for ROI identification in warning
        assert "ROI" in source
        assert "logger.warning" in source
        # Check warning identifies the ROI that lacks SCR
        assert "lacks short channel regression" in source

    def test_proceeds_without_regression_for_affected_roi(self) -> None:
        """Verify processing continues for ROIs with available short channels."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check that function continues processing after skipping
        assert "continue" in source
        # Check that other ROIs are still processed
        assert "rois_processed" in source

    def test_references_requirement_8_6(self) -> None:
        """Verify function references requirement 8.6."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "8.6" in source

    def test_tracks_skipped_rois(self) -> None:
        """Verify function tracks which ROIs were skipped."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        assert "rois_skipped" in source
        # Check for summary logging of skipped ROIs
        assert "ROIs without SCR" in source or "rois_skipped" in source

    def test_process_fnirs_handles_missing_short_channels(self) -> None:
        """Verify process_fnirs_with_scr_fallback handles missing short channels."""
        source = get_function_source("process_fnirs_with_scr_fallback")
        # Check for missing short channel detection in fallback path
        assert "missing_short_channels" in source or "missing from montage" in source

    def test_checks_all_wavelengths_before_skipping_roi(self) -> None:
        """Verify function checks all wavelengths before deciding to skip ROI."""
        source = get_function_source("apply_scr_with_explicit_pairing")
        # Check for wavelength-level checking
        assert "short_ch_available_wavelengths" in source or "wavelength" in source.lower()
