"""
Test suite for generate_visualizations function.

This module tests the visualization generation functionality of the unified
analysis pipeline, ensuring that visualizations are created correctly for
EEG and fNIRS data.
"""

import sys
from pathlib import Path
import tempfile

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_analysis import generate_visualizations
from affective_fnirs.config import (
    SubjectConfig,
    SubjectInfo,
    ModalityConfig,
    ReportConfig,
    TrialsConfig,
    FilterConfig,
    QualityThresholds,
    EpochConfig,
    AnalysisConfig,
    ICAConfig,
    MotionCorrectionConfig,
)


def create_minimal_config():
    """Create a minimal SubjectConfig for testing."""
    return SubjectConfig(
        subject=SubjectInfo(id="002", session="001", task="fingertapping"),
        modalities=ModalityConfig(eeg_enabled=True, fnirs_enabled=True),
        report=ReportConfig(qa_only=False),
        eeg_channels_of_interest=["C3", "C4"],
        trials=TrialsConfig(count_per_condition=10, task_duration_sec=15.0, rest_duration_sec=15.0),
        filters=FilterConfig(
            eeg_bandpass_low_hz=1.0,
            eeg_bandpass_high_hz=40.0,
            fnirs_bandpass_low_hz=0.01,
            fnirs_bandpass_high_hz=0.5,
            cardiac_band_low_hz=0.5,
            cardiac_band_high_hz=2.5,
        ),
        quality=QualityThresholds(
            sci_threshold=0.75,
            cv_threshold_percent=15.0,
            saturation_percent=5.0,
            psp_threshold=0.1,
            short_channel_distance_mm=15.0,
        ),
        epochs=EpochConfig(
            eeg_tmin_sec=-3.0,
            eeg_tmax_sec=20.0,
            fnirs_tmin_sec=-3.0,
            fnirs_tmax_sec=30.0,
            baseline_tmin_sec=-3.0,
            baseline_tmax_sec=-1.0,
        ),
        analysis=AnalysisConfig(
            alpha_band_low_hz=8.0,
            alpha_band_high_hz=13.0,
            beta_band_low_hz=13.0,
            beta_band_high_hz=30.0,
            baseline_window_start_sec=-3.0,
            baseline_window_end_sec=-1.0,
            task_window_start_sec=1.0,
            task_window_end_sec=14.0,
            beta_rebound_window_start_sec=15.0,
            beta_rebound_window_end_sec=20.0,
            hrf_onset_window_start_sec=2.0,
            hrf_onset_window_end_sec=3.0,
            hrf_peak_window_start_sec=4.0,
            hrf_peak_window_end_sec=8.0,
            dpf=6.0,
        ),
        ica=ICAConfig(enabled=False, n_components=15, random_state=42, max_iter=200),
        motion_correction=MotionCorrectionConfig(method="tddr"),
        data_root=Path("data/raw"),
        output_root=Path("data/derivatives/validation-pipeline"),
        random_seed=42,
    )


def test_generate_visualizations_no_results():
    """
    Test that generate_visualizations handles None results gracefully.
    
    When no analysis results are available (both EEG and fNIRS are None),
    the function should return an empty dictionary and not raise errors.
    """
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        
        # Create minimal config
        config = create_minimal_config()
        
        # Call with None results
        viz_paths = generate_visualizations(
            eeg_results=None,
            fnirs_results=None,
            config=config,
            output_path=output_path,
        )
        
        # Should return empty dict
        assert isinstance(viz_paths, dict)
        assert len(viz_paths) == 0
        print("✓ Test passed: generate_visualizations handles None results correctly")


def test_generate_visualizations_creates_output_dir():
    """
    Test that generate_visualizations creates output directory if it doesn't exist.
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nonexistent" / "subdir"
        
        # Verify directory doesn't exist
        assert not output_path.exists()
        
        # Create minimal config
        config = create_minimal_config()
        
        # Call with None results (should still create directory)
        viz_paths = generate_visualizations(
            eeg_results=None,
            fnirs_results=None,
            config=config,
            output_path=output_path,
        )
        
        # Directory should now exist
        assert output_path.exists()
        assert output_path.is_dir()
        print("✓ Test passed: generate_visualizations creates output directory")


if __name__ == "__main__":
    print("Running generate_visualizations tests...")
    print()
    
    test_generate_visualizations_no_results()
    test_generate_visualizations_creates_output_dir()
    
    print()
    print("All tests passed!")

