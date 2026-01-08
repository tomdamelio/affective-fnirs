"""
affective_fnirs: Multimodal EEG + fNIRS Validation Pipeline.

This package provides tools for processing and validating simultaneous
EEG and fNIRS recordings during finger tapping tasks, including signal
quality assessment, ERD/ERS detection, HRF analysis, and neurovascular
coupling quantification.
"""

from affective_fnirs.bids_utils import (
    BIDSValidationError,
    RawDataWriteError,
    generate_derivative_path,
    validate_bids_path,
    validate_file_mode,
)
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
    compute_tfr_by_condition,
    create_epochs,
    define_motor_roi_clusters,
    detect_erd_ers,
    plot_erd_timecourse,
    plot_eeg_spectrogram,
    plot_spectrogram_by_condition,
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
from affective_fnirs.lateralization_analysis import (
    LateralizationResult,
    LateralizationTimeCourseData,
    compute_lateralization_analysis,
    compute_lateralization_timecourse,
    plot_lateralization_barplot,
    plot_lateralization_results,
    plot_lateralization_timecourse,
)
from affective_fnirs.pipeline import run_validation_pipeline

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
    # BIDS Utilities
    "validate_bids_path",
    "generate_derivative_path",
    "validate_file_mode",
    "BIDSValidationError",
    "RawDataWriteError",
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
    "compute_tfr_by_condition",
    "select_motor_channel",
    "detect_erd_ers",
    "define_motor_roi_clusters",
    "plot_eeg_spectrogram",
    "plot_spectrogram_by_condition",
    "plot_erd_timecourse",
    # fNIRS Analysis
    "create_fnirs_epochs",
    "identify_motor_roi_channel",
    "extract_hrf",
    "validate_hrf_temporal_dynamics",
    "compute_hrf_quality_metrics",
    "plot_hrf_curves",
    "plot_hrf_spatial_map",
    # Lateralization Analysis
    "LateralizationResult",
    "LateralizationTimeCourseData",
    "compute_lateralization_analysis",
    "compute_lateralization_timecourse",
    "plot_lateralization_barplot",
    "plot_lateralization_results",
    "plot_lateralization_timecourse",
    # Pipeline
    "run_validation_pipeline",
]
