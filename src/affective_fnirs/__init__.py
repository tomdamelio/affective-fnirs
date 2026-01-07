"""
affective_fnirs: Multimodal EEG + fNIRS Validation Pipeline.

This package provides tools for processing and validating simultaneous
EEG and fNIRS recordings during finger tapping tasks, including signal
quality assessment, ERD/ERS detection, HRF analysis, and neurovascular
coupling quantification.
"""

from affective_fnirs.config import (
    AnalysisConfig,
    EpochConfig,
    FilterConfig,
    ICAConfig,
    MotionCorrectionConfig,
    PipelineConfig,
    QualityThresholds,
)
from affective_fnirs.eeg_analysis import (
    compute_tfr,
    create_epochs,
    detect_erd_ers,
    plot_erd_timecourse,
    plot_eeg_spectrogram,
    select_motor_channel,
)
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
from affective_fnirs.fnirs_analysis import (
    compute_hrf_quality_metrics,
    create_fnirs_epochs,
    extract_hrf,
    identify_motor_roi_channel,
    plot_hrf_curves,
    plot_hrf_spatial_map,
    validate_hrf_temporal_dynamics,
)

__version__ = "0.1.0"
__author__ = "Sebastian et al."

# Public API
__all__ = [
    # Configuration
    "PipelineConfig",
    "FilterConfig",
    "QualityThresholds",
    "EpochConfig",
    "AnalysisConfig",
    "ICAConfig",
    "MotionCorrectionConfig",
    # EEG Processing
    "preprocess_eeg",
    "detect_bad_eeg_channels",
    "apply_ica_artifact_removal",
    "identify_eog_components",
    "identify_emg_components",
    "interpolate_bad_channels",
    "rereference_eeg",
    "preprocess_eeg_pipeline",
    # EEG Analysis
    "create_epochs",
    "compute_tfr",
    "select_motor_channel",
    "detect_erd_ers",
    "plot_eeg_spectrogram",
    "plot_erd_timecourse",
    # fNIRS Analysis
    "create_fnirs_epochs",
    "identify_motor_roi_channel",
    "extract_hrf",
    "validate_hrf_temporal_dynamics",
    "compute_hrf_quality_metrics",
    "plot_hrf_curves",
    "plot_hrf_spatial_map",
]
