"""
Main Pipeline Orchestration for Multimodal Validation Pipeline.

This module provides the main pipeline runner that orchestrates all stages
of the validation pipeline for simultaneous EEG + fNIRS recordings during
finger tapping tasks.

Pipeline Stages:
    1. Load XDF and JSON metadata
    2. Build MNE objects (EEG and fNIRS separately)
    3. fNIRS quality assessment (on raw intensity)
    4. fNIRS processing (OD → TDDR → SCR → Hb → Filter)
    5. EEG preprocessing (Filter → ICA → Interpolate → CAR)
    6. EEG analysis (Epochs → TFR → ERD/ERS)
    7. fNIRS analysis (Epochs → HRF → Validation)
    8. Multimodal coupling (Envelope → Cross-correlation)
    9. Generate reports (Quality, HTML, JSON)

Requirements:
    - All requirements (comprehensive pipeline)
    - 11.6: Progress logging for long operations
    - 10.2, 10.3: Reproducibility with seed control

References:
    - MNE-Python: https://mne.tools/stable/
    - MNE-NIRS: https://mne.tools/mne-nirs/stable/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np

from affective_fnirs.bids_utils import generate_derivative_path, validate_bids_path
from affective_fnirs.config import PipelineConfig
from affective_fnirs.eeg_analysis import (
    compute_tfr,
    compute_tfr_by_condition,
    create_epochs,
    define_motor_roi_clusters,
    detect_erd_ers,
    plot_condition_contrast_spectrograms,
    plot_erd_timecourse,
    plot_erd_timecourse_bilateral,
    plot_eeg_spectrogram,
    plot_spectrogram_by_condition,
    select_motor_channel,
)
from affective_fnirs.eeg_processing import preprocess_eeg_pipeline
from affective_fnirs.fnirs_analysis import (
    compute_hrf_quality_metrics,
    create_fnirs_epochs,
    extract_hrf,
    identify_motor_roi_channel,
    plot_hrf_curves,
    plot_hrf_spatial_map,
    validate_hrf_temporal_dynamics,
)
from affective_fnirs.fnirs_processing import process_fnirs_pipeline
from affective_fnirs.fnirs_quality import (
    assess_cardiac_power,
    calculate_coefficient_of_variation,
    calculate_sci,
    detect_saturation,
    generate_quality_heatmap,
    mark_bad_channels,
)
from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.lateralization_analysis import (
    LateralizationResult,
    compute_lateralization_analysis,
    compute_lateralization_timecourse,
    plot_lateralization_barplot,
    plot_lateralization_results,
    plot_lateralization_timecourse,
)
from affective_fnirs.mne_builder import build_eeg_raw, build_fnirs_raw, embed_events
from affective_fnirs.multimodal_analysis import (
    compute_neurovascular_coupling,
    extract_eeg_envelope,
    plot_coupling_overlay,
    resample_to_fnirs,
)
from affective_fnirs.reporting import (
    ChannelQuality,
    CouplingMetrics,
    ERDMetrics,
    HRFValidation,
    LateralizationMetrics,
    QualityReport,
    ValidationResults,
    generate_quality_report,
    generate_validation_report_html,
    save_numerical_results,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Exception raised for pipeline execution failures."""

    pass


def run_validation_pipeline(
    xdf_file: Path,
    eeg_json: Path,
    fnirs_json: Path,
    config: PipelineConfig | None = None,
    output_dir: Path | None = None,
) -> ValidationResults:
    """
    Run complete validation pipeline on multimodal EEG + fNIRS data.

    This is the main entry point for the validation pipeline. It orchestrates
    all processing stages from raw XDF data to final validation reports.

    Pipeline Stages:
        1. Load XDF and JSON metadata
        2. Build MNE objects (EEG and fNIRS separately)
        3. fNIRS quality assessment (on raw intensity)
        4. fNIRS processing (OD → TDDR → SCR → Hb → Filter)
        5. EEG preprocessing (Filter → ICA → Interpolate → CAR)
        6. EEG analysis (Epochs → TFR → ERD/ERS)
        7. fNIRS analysis (Epochs → HRF → Validation)
        8. Multimodal coupling (Envelope → Cross-correlation)
        9. Generate reports (Quality, HTML, JSON)

    Args:
        xdf_file: Path to XDF file containing EEG, fNIRS, and marker streams
        eeg_json: Path to EEG JSON sidecar with channel metadata
        fnirs_json: Path to fNIRS JSON sidecar with montage configuration
        config: Pipeline configuration (uses defaults if None)
        output_dir: Output directory for derivatives (uses config if None)

    Returns:
        ValidationResults dataclass with all metrics and validation outcomes

    Raises:
        PipelineError: If any stage fails with detailed error message
        FileNotFoundError: If input files don't exist
        ValueError: If BIDS paths are invalid

    Requirements:
        - All requirements (comprehensive pipeline)
        - 11.6: Progress logging for long operations
        - 10.2, 10.3: Reproducibility with seed control

    Example:
        >>> from pathlib import Path
        >>> from affective_fnirs.config import PipelineConfig
        >>> from affective_fnirs.pipeline import run_validation_pipeline
        >>>
        >>> xdf_file = Path("data/raw/sub-002/sub-002_recording.xdf")
        >>> eeg_json = Path("data/raw/sub-002/sub-002_eeg.json")
        >>> fnirs_json = Path("data/raw/sub-002/sub-002_fnirs.json")
        >>> config = PipelineConfig.default()
        >>>
        >>> results = run_validation_pipeline(xdf_file, eeg_json, fnirs_json, config)
        >>> print(f"EEG ERD: {results.erd_metrics.alpha_erd_percent:.1f}%")
    """
    # Initialize configuration
    if config is None:
        config = PipelineConfig.default()
        logger.info("Using default pipeline configuration")
    else:
        logger.info("Using provided pipeline configuration")

    # Set random seed for reproducibility (Req. 10.2)
    np.random.seed(config.random_seed)
    logger.info(f"Random seed set to: {config.random_seed}")

    # Validate input paths
    if not xdf_file.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_file}")
    if not eeg_json.exists():
        raise FileNotFoundError(f"EEG JSON not found: {eeg_json}")
    if not fnirs_json.exists():
        raise FileNotFoundError(f"fNIRS JSON not found: {fnirs_json}")

    # Extract subject/session/task from filename (BIDS naming)
    try:
        validate_bids_path(xdf_file)
        filename_parts = xdf_file.stem.split("_")
        subject_id = None
        session_id = None
        task = None

        for part in filename_parts:
            if part.startswith("sub-"):
                subject_id = part.replace("sub-", "")
            elif part.startswith("ses-"):
                session_id = part.replace("ses-", "")
            elif part.startswith("task-"):
                task = part.replace("task-", "")

        if not subject_id:
            raise ValueError("Could not extract subject ID from filename")
        if not session_id:
            session_id = "001"  # Default session
        if not task:
            task = "fingertapping"  # Default task

        logger.info(f"Processing: sub-{subject_id}, ses-{session_id}, task-{task}")
    except Exception as e:
        raise PipelineError(
            f"Failed to parse BIDS filename: {xdf_file.name}\n"
            f"Expected format: sub-XX_ses-XX_task-XX_...\n"
            f"Error: {e}"
        ) from e

    # Determine output directory
    if output_dir is None:
        output_dir = generate_derivative_path(
            config.output_root, subject_id, session_id
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get current timestamp
    timestamp = datetime.now().isoformat()

    # Get software versions (Req. 10.3)
    software_versions = {
        "mne": mne.__version__,
        "numpy": np.__version__,
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
    }


    # =========================================================================
    # STAGE 1: Load XDF and JSON metadata
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 1: Loading XDF file and JSON metadata")
    logger.info("=" * 70)

    try:
        # Load XDF file
        logger.info(f"Loading XDF file: {xdf_file.name}")
        streams, header = load_xdf_file(xdf_file)
        logger.info(f"Loaded {len(streams)} streams from XDF")

        # Identify streams
        logger.info("Identifying EEG, fNIRS, and Marker streams...")
        identified_streams = identify_streams(streams)
        logger.info(
            f"Identified streams: {', '.join(identified_streams.keys()).upper()}"
        )

        # Extract stream data
        logger.info("Extracting EEG stream data...")
        eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(
            identified_streams["eeg"]
        )
        logger.info(
            f"EEG: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels, {eeg_sfreq} Hz"
        )

        logger.info("Extracting fNIRS stream data...")
        fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(
            identified_streams["fnirs"]
        )
        logger.info(
            f"fNIRS: {fnirs_data.shape[0]} samples, {fnirs_data.shape[1]} channels, {fnirs_sfreq} Hz"
        )

        logger.info("Extracting marker stream data...")
        marker_stream = identified_streams["markers"]

        # Load JSON metadata
        logger.info(f"Loading EEG JSON: {eeg_json.name}")
        with open(eeg_json, "r") as f:
            eeg_metadata = json.load(f)

        logger.info(f"Loading fNIRS JSON: {fnirs_json.name}")
        with open(fnirs_json, "r") as f:
            fnirs_metadata = json.load(f)

    except Exception as e:
        raise PipelineError(f"Stage 1 failed (Data Ingestion): {e}") from e


    # =========================================================================
    # STAGE 2: Build MNE objects (EEG and fNIRS separately)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 2: Building MNE Raw objects")
    logger.info("=" * 70)

    try:
        # Build EEG Raw object
        logger.info("Building EEG Raw object with 10-20 montage...")
        raw_eeg = build_eeg_raw(
            eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
        )
        logger.info(f"EEG Raw: {len(raw_eeg.ch_names)} channels, {raw_eeg.times[-1]:.1f}s duration")

        # Build fNIRS Raw object
        logger.info("Building fNIRS Raw object with wavelength metadata...")
        fnirs_montage = fnirs_metadata.get("ChMontage", [])
        raw_fnirs = build_fnirs_raw(
            fnirs_data, fnirs_sfreq, fnirs_montage, fnirs_timestamps
        )
        logger.info(
            f"fNIRS Raw: {len(raw_fnirs.ch_names)} channels, {raw_fnirs.times[-1]:.1f}s duration"
        )

        # Embed event markers
        logger.info("Embedding event markers with LSL timestamps...")
        event_mapping = {
            "LEFT": 1,
            "RIGHT": 2,
            "NOTHING": 3,
            "task_start": 10,
            "task_end": 11,
        }
        raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)
        raw_fnirs = embed_events(raw_fnirs, marker_stream, event_mapping)
        logger.info(f"Embedded {len(raw_eeg.annotations)} events")

    except Exception as e:
        raise PipelineError(f"Stage 2 failed (MNE Construction): {e}") from e


    # =========================================================================
    # STAGE 3: fNIRS quality assessment (on raw intensity)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 3: fNIRS Quality Assessment")
    logger.info("=" * 70)

    try:
        # Calculate quality metrics on raw intensity
        logger.info("Calculating Scalp Coupling Index (SCI)...")
        sci_values = calculate_sci(
            raw_fnirs.copy(),
            freq_range=(
                config.filters.cardiac_band_low_hz,
                config.filters.cardiac_band_high_hz,
            ),
            sci_threshold=config.quality.sci_threshold,
        )
        mean_sci = np.mean(list(sci_values.values()))
        logger.info(f"Mean SCI: {mean_sci:.3f}")

        logger.info("Detecting signal saturation...")
        saturation_values = detect_saturation(
            raw_fnirs.copy(),
            saturation_threshold=0.95,
            max_saturation_percent=config.quality.saturation_percent,
        )
        mean_saturation = np.mean(list(saturation_values.values()))
        logger.info(f"Mean saturation: {mean_saturation:.2f}%")

        logger.info("Assessing cardiac pulsation power...")
        cardiac_power = assess_cardiac_power(
            raw_fnirs.copy(),
            freq_range=(
                config.filters.cardiac_band_low_hz,
                config.filters.cardiac_band_high_hz,
            ),
            power_threshold=config.quality.psp_threshold,
        )
        mean_cardiac = np.mean(list(cardiac_power.values()))
        logger.info(f"Mean cardiac power: {mean_cardiac:.3f}")

        # Calculate CV on baseline periods
        logger.info("Calculating Coefficient of Variation (CV) on baseline...")
        # Extract baseline annotations (5s before each event)
        baseline_annotations = []
        for onset in raw_fnirs.annotations.onset:
            if onset >= 5.0:  # Ensure we have 5s before
                baseline_annotations.append((onset - 5.0, onset))

        cv_values = calculate_coefficient_of_variation(
            raw_fnirs.copy(),
            baseline_annotations,
            cv_threshold=config.quality.cv_threshold_percent,
        )
        mean_cv = np.mean(list(cv_values.values()))
        logger.info(f"Mean CV: {mean_cv:.2f}%")

        # Mark bad channels
        logger.info("Marking bad channels based on quality thresholds...")
        raw_fnirs, failure_reasons = mark_bad_channels(
            raw_fnirs,
            sci_values,
            saturation_values,
            cardiac_power,
            cv_values,
            sci_threshold=config.quality.sci_threshold,
            saturation_threshold=config.quality.saturation_percent,
            psp_threshold=config.quality.psp_threshold,
            cv_threshold=config.quality.cv_threshold_percent,
        )
        n_bad = len(raw_fnirs.info["bads"])
        n_total = len(raw_fnirs.ch_names)
        logger.info(f"Marked {n_bad}/{n_total} channels as bad")

        # Generate quality heatmap
        logger.info("Generating quality heatmap...")
        quality_heatmap = generate_quality_heatmap(
            raw_fnirs, 
            sci_values, 
            saturation_values,
            cardiac_power,
            cv_values,
            failure_reasons
        )

        # Create quality report dataclass
        channel_qualities = []
        for ch_name in raw_fnirs.ch_names:
            is_bad = ch_name in raw_fnirs.info["bads"]
            reason = "; ".join(failure_reasons.get(ch_name, [])) if is_bad else "Good"

            channel_qualities.append(
                ChannelQuality(
                    channel_name=ch_name,
                    sci=sci_values.get(ch_name, 0.0),
                    saturation_percent=saturation_values.get(ch_name, 0.0),
                    cardiac_power=cardiac_power.get(ch_name, 0.0),
                    cv=cv_values.get(ch_name, 0.0),
                    is_bad=is_bad,
                    reason=reason,
                )
            )

        quality_report = QualityReport(
            channels=channel_qualities,
            n_total_channels=n_total,
            n_bad_channels=n_bad,
            mean_sci=mean_sci,
            mean_saturation=mean_saturation,
            mean_cardiac_power=mean_cardiac,
            mean_cv=mean_cv,
        )

    except Exception as e:
        raise PipelineError(f"Stage 3 failed (fNIRS Quality): {e}") from e


    # =========================================================================
    # STAGE 4: fNIRS processing (OD → TDDR → SCR → Hb → Filter)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 4: fNIRS Processing Pipeline")
    logger.info("=" * 70)

    try:
        logger.info("Running fNIRS processing pipeline...")
        logger.info("  1. Intensity → Optical Density")
        logger.info("  2. Motion correction (TDDR)")
        logger.info("  3. Short channel regression")
        logger.info("  4. OD → Hemoglobin (Beer-Lambert)")
        logger.info("  5. Bandpass filter (0.01-0.5 Hz)")

        raw_fnirs_processed, fnirs_processing_metrics = process_fnirs_pipeline(
            raw_fnirs.copy(), 
            fnirs_metadata,  # Pass full metadata dict (contains ChMontage)
            motion_correction_method=config.motion_correction.method,
            dpf=6.0,  # Default DPF for adults
            l_freq=config.filters.fnirs_bandpass_low_hz,
            h_freq=config.filters.fnirs_bandpass_high_hz,
            short_threshold_mm=config.quality.short_channel_distance_mm,
            apply_scr=True,  # Always apply short channel regression
            verify_noise_reduction=True,  # Always verify noise reduction
        )
        logger.info("fNIRS processing complete")

    except Exception as e:
        raise PipelineError(f"Stage 4 failed (fNIRS Processing): {e}") from e


    # =========================================================================
    # STAGE 5: EEG preprocessing (Filter → ICA → Interpolate → CAR)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 5: EEG Preprocessing Pipeline")
    logger.info("=" * 70)

    try:
        logger.info("Running EEG preprocessing pipeline...")
        logger.info("  1. Bandpass filter (1-40 Hz)")
        logger.info("  2. Detect bad channels")
        logger.info("  3. Fit ICA")
        logger.info("  4. Identify artifact components (EOG, EMG)")
        logger.info("  5. Apply ICA")
        logger.info("  6. Interpolate bad channels")
        logger.info("  7. Common Average Reference")

        raw_eeg_processed, ica = preprocess_eeg_pipeline(
            raw_eeg.copy(), config
        )
        logger.info("EEG preprocessing complete")

        # Save ICA for reproducibility (if ICA was applied)
        if ica is not None:
            ica_path = output_dir / f"sub-{subject_id}_ses-{session_id}_task-{task}_ica.fif"
            ica.save(ica_path, overwrite=True)
            logger.info(f"Saved ICA to: {ica_path.name}")
        else:
            logger.info("ICA was skipped - no ICA file to save")

    except Exception as e:
        raise PipelineError(f"Stage 5 failed (EEG Preprocessing): {e}") from e


    # =========================================================================
    # STAGE 6: EEG analysis (Epochs → TFR → ERD/ERS)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 6: EEG ERD/ERS Analysis")
    logger.info("=" * 70)

    try:
        # Create epochs
        logger.info("Creating EEG epochs...")
        events, event_id = mne.events_from_annotations(raw_eeg_processed)
        epochs_eeg = create_epochs(
            raw_eeg_processed,
            event_id,
            tmin=config.epochs.eeg_tmin_sec,
            tmax=config.epochs.eeg_tmax_sec,
            baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
        )
        logger.info(f"Created {len(epochs_eeg)} EEG epochs")

        # Select motor channel
        logger.info("Selecting motor cortex channel...")
        motor_channel = select_motor_channel(epochs_eeg)
        logger.info(f"Selected channel: {motor_channel}")

        # Compute TFR (all conditions combined)
        logger.info("Computing time-frequency representation (all conditions)...")
        tfr = compute_tfr(
            epochs_eeg,
            freqs=np.arange(3, 31, 1),
            n_cycles=7,
            baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
            baseline_mode="percent",
        )
        logger.info("TFR computation complete")

        # Detect ERD/ERS for both C3 and C4
        logger.info("Detecting ERD/ERS patterns for C3 and C4...")
        
        # C3 (left motor cortex)
        erd_results_c3 = detect_erd_ers(
            tfr,
            "C3",
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
            task_window=(
                config.analysis.task_window_start_sec,
                config.analysis.task_window_end_sec,
            ),
            baseline_window=(
                config.analysis.baseline_window_start_sec,
                config.analysis.baseline_window_end_sec,
            ),
            beta_rebound_window=(
                config.analysis.beta_rebound_window_start_sec,
                config.analysis.beta_rebound_window_end_sec,
            ),
        )
        
        # C4 (right motor cortex)
        erd_results_c4 = detect_erd_ers(
            tfr,
            "C4",
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
            task_window=(
                config.analysis.task_window_start_sec,
                config.analysis.task_window_end_sec,
            ),
            baseline_window=(
                config.analysis.baseline_window_start_sec,
                config.analysis.baseline_window_end_sec,
            ),
            beta_rebound_window=(
                config.analysis.beta_rebound_window_start_sec,
                config.analysis.beta_rebound_window_end_sec,
            ),
        )
        
        logger.info(
            f"C3 - Alpha ERD: {erd_results_c3['alpha_erd_percent']:.1f}%, Beta ERD: {erd_results_c3['beta_erd_percent']:.1f}%"
        )
        logger.info(
            f"C4 - Alpha ERD: {erd_results_c4['alpha_erd_percent']:.1f}%, Beta ERD: {erd_results_c4['beta_erd_percent']:.1f}%"
        )

        # Create ERD metrics dataclass (keep C3 as primary for backward compatibility)
        erd_metrics = ERDMetrics(
            channel=motor_channel,
            alpha_erd_percent=erd_results_c3["alpha_erd_percent"],
            alpha_p_value=erd_results_c3["alpha_p_value"],
            alpha_significant=erd_results_c3["alpha_significant"],
            beta_erd_percent=erd_results_c3["beta_erd_percent"],
            beta_p_value=erd_results_c3["beta_p_value"],
            beta_significant=erd_results_c3["beta_significant"],
            beta_rebound_percent=erd_results_c3.get("beta_rebound_percent", np.nan),
            beta_rebound_observed=erd_results_c3.get("beta_rebound_observed", False),
        )
        
        # Store C4 results separately for reporting
        erd_metrics_c4 = ERDMetrics(
            channel="C4",
            alpha_erd_percent=erd_results_c4["alpha_erd_percent"],
            alpha_p_value=erd_results_c4["alpha_p_value"],
            alpha_significant=erd_results_c4["alpha_significant"],
            beta_erd_percent=erd_results_c4["beta_erd_percent"],
            beta_p_value=erd_results_c4["beta_p_value"],
            beta_significant=erd_results_c4["beta_significant"],
            beta_rebound_percent=erd_results_c4.get("beta_rebound_percent", np.nan),
            beta_rebound_observed=erd_results_c4.get("beta_rebound_observed", False),
        )

        # Generate bilateral ERD timecourse (C3 and C4)
        logger.info("Generating bilateral ERD timecourse (C3 and C4)...")
        erd_timecourse = plot_erd_timecourse_bilateral(
            tfr,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
            task_onset=0.0,
            task_offset=config.analysis.task_window_end_sec,
        )

        # Compute TFR by condition for detailed analysis
        logger.info("Computing TFR by condition (LEFT, RIGHT, NOTHING)...")
        tfr_by_condition = compute_tfr_by_condition(
            raw_eeg_processed,
            freqs=np.arange(3, 31, 1),
            n_cycles=7,
            tmin=config.epochs.eeg_tmin_sec,
            tmax=config.epochs.eeg_tmax_sec,
            baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
            baseline_mode="percent",
        )

        # Define motor ROI clusters
        motor_clusters = define_motor_roi_clusters()
        logger.info(f"Motor ROI clusters defined: {list(motor_clusters.keys())}")

        # Generate condition contrast spectrograms (LEFT-RIGHT for left cluster, RIGHT-LEFT for right cluster)
        logger.info("Generating condition contrast spectrograms...")
        eeg_spectrogram = plot_condition_contrast_spectrograms(
            tfr_by_condition,
            motor_clusters,
            vmin=-50.0,
            vmax=50.0,
            task_onset=0.0,
            task_offset=config.analysis.task_window_end_sec,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
        )

        # Generate spectrograms by condition for both hemispheres (for detailed inspection)
        logger.info("Generating spectrograms by condition for left motor cortex...")
        eeg_spectrogram_left_by_condition = plot_spectrogram_by_condition(
            tfr_by_condition,
            roi_name="Left Motor Cortex (C3 cluster)",
            channels=motor_clusters['left_motor'],
            vmin=-50.0,
            vmax=50.0,
            task_onset=0.0,
            task_offset=config.analysis.task_window_end_sec,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
        )

        logger.info("Generating spectrograms by condition for right motor cortex...")
        eeg_spectrogram_right_by_condition = plot_spectrogram_by_condition(
            tfr_by_condition,
            roi_name="Right Motor Cortex (C4 cluster)",
            channels=motor_clusters['right_motor'],
            vmin=-50.0,
            vmax=50.0,
            task_onset=0.0,
            task_offset=config.analysis.task_window_end_sec,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
        )

        logger.info("EEG analysis complete with condition contrast and detailed spectrograms")

    except Exception as e:
        raise PipelineError(f"Stage 6 failed (EEG Analysis): {e}") from e


    # =========================================================================
    # STAGE 6.5: Lateralization Analysis (LEFT vs RIGHT vs NOTHING)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 6.5: Lateralization Analysis")
    logger.info("=" * 70)

    lateralization_result = None
    lateralization_metrics = None
    lateralization_timecourse_fig = None
    lateralization_barplot_fig = None

    try:
        logger.info("Computing lateralization analysis for motor conditions...")
        logger.info("  Expected: LEFT hand → C4 ERD (contralateral)")
        logger.info("  Expected: RIGHT hand → C3 ERD (contralateral)")
        logger.info("  Expected: NOTHING → No significant ERD")

        # Create filtered-only EEG for lateralization analysis
        # Note: CAR distorts ERD patterns, so we use filtered data without CAR
        logger.info("  Using filtered EEG (without CAR) for accurate ERD detection")
        raw_eeg_filtered = raw_eeg.copy()
        raw_eeg_filtered.filter(
            l_freq=config.filters.eeg_bandpass_low_hz,
            h_freq=config.filters.eeg_bandpass_high_hz,
            verbose=False,
        )

        # Run lateralization analysis on filtered EEG (without CAR)
        lateralization_result = compute_lateralization_analysis(
            raw_eeg_filtered,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
            tmin=config.epochs.eeg_tmin_sec,
            tmax=config.epochs.eeg_tmax_sec,
            baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
            task_window=(
                config.analysis.task_window_start_sec,
                config.analysis.task_window_end_sec,
            ),
        )

        # Convert to LateralizationMetrics for reporting
        lateralization_metrics = LateralizationMetrics(
            left_c3_alpha_erd=lateralization_result.left_c3_alpha_erd,
            left_c4_alpha_erd=lateralization_result.left_c4_alpha_erd,
            right_c3_alpha_erd=lateralization_result.right_c3_alpha_erd,
            right_c4_alpha_erd=lateralization_result.right_c4_alpha_erd,
            nothing_c3_alpha_erd=lateralization_result.nothing_c3_alpha_erd,
            nothing_c4_alpha_erd=lateralization_result.nothing_c4_alpha_erd,
            left_c3_beta_erd=lateralization_result.left_c3_beta_erd,
            left_c4_beta_erd=lateralization_result.left_c4_beta_erd,
            right_c3_beta_erd=lateralization_result.right_c3_beta_erd,
            right_c4_beta_erd=lateralization_result.right_c4_beta_erd,
            nothing_c3_beta_erd=lateralization_result.nothing_c3_beta_erd,
            nothing_c4_beta_erd=lateralization_result.nothing_c4_beta_erd,
            left_lateralization_alpha=lateralization_result.left_lateralization_alpha,
            right_lateralization_alpha=lateralization_result.right_lateralization_alpha,
            left_lateralization_beta=lateralization_result.left_lateralization_beta,
            right_lateralization_beta=lateralization_result.right_lateralization_beta,
            left_vs_nothing_c4_p=lateralization_result.left_vs_nothing_c4_p,
            right_vs_nothing_c3_p=lateralization_result.right_vs_nothing_c3_p,
            left_contralateral_vs_ipsilateral_p=lateralization_result.left_contralateral_vs_ipsilateral_p,
            right_contralateral_vs_ipsilateral_p=lateralization_result.right_contralateral_vs_ipsilateral_p,
            left_shows_contralateral_erd=lateralization_result.left_shows_contralateral_erd,
            right_shows_contralateral_erd=lateralization_result.right_shows_contralateral_erd,
            lateralization_pattern_valid=lateralization_result.lateralization_pattern_valid,
            n_left_trials=lateralization_result.n_left_trials,
            n_right_trials=lateralization_result.n_right_trials,
            n_nothing_trials=lateralization_result.n_nothing_trials,
        )

        # Compute time-course data for lineplots
        logger.info("Computing lateralization time-course...")
        lateralization_timecourse_data = compute_lateralization_timecourse(
            raw_eeg_filtered,
            alpha_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
            beta_band=(config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz),
            tmin=config.epochs.eeg_tmin_sec,
            tmax=config.epochs.eeg_tmax_sec,
            baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
        )

        # Generate lateralization visualizations for HTML report
        logger.info("Generating lateralization visualizations...")
        lateralization_timecourse_fig = plot_lateralization_timecourse(
            lateralization_timecourse_data,
            lateralization_result,
        )
        lateralization_barplot_fig = plot_lateralization_barplot(lateralization_result)
        logger.info("Lateralization figures generated for HTML report")

    except Exception as e:
        logger.warning(f"Stage 6.5 failed (Lateralization Analysis): {e}")
        logger.warning("Continuing without lateralization analysis...")


    # =========================================================================
    # STAGE 7: fNIRS analysis (Epochs → HRF → Validation)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 7: fNIRS HRF Analysis")
    logger.info("=" * 70)

    try:
        # Create fNIRS epochs
        logger.info("Creating fNIRS epochs...")
        # Note: create_fnirs_epochs extracts events internally from annotations
        _, event_id_fnirs = mne.events_from_annotations(raw_fnirs_processed)
        epochs_fnirs = create_fnirs_epochs(
            raw_fnirs_processed,
            event_id=event_id_fnirs,
            tmin=config.epochs.fnirs_tmin_sec,
            tmax=config.epochs.fnirs_tmax_sec,
            baseline=(config.epochs.baseline_tmin_sec, 0.0),
        )
        logger.info(f"Created {len(epochs_fnirs)} fNIRS epochs")

        # Identify motor ROI channel
        logger.info("Identifying motor ROI channel...")
        motor_roi_channel = identify_motor_roi_channel(epochs_fnirs, motor_channel)
        logger.info(f"Selected fNIRS channel: {motor_roi_channel}")

        # Extract HRF
        logger.info("Extracting hemodynamic response function...")
        # Extract HbO HRF
        times, hrf_hbo = extract_hrf(epochs_fnirs, motor_roi_channel, chromophore="hbo")
        # Extract HbR HRF (find corresponding HbR channel)
        hbr_channel = motor_roi_channel.replace("hbo", "hbr")
        if hbr_channel in epochs_fnirs.ch_names:
            _, hrf_hbr = extract_hrf(epochs_fnirs, hbr_channel, chromophore="hbr")
        else:
            logger.warning(f"HbR channel {hbr_channel} not found, using zeros")
            hrf_hbr = np.zeros_like(hrf_hbo)
        logger.info(f"HRF extracted: {len(times)} time points")

        # Validate HRF temporal dynamics
        logger.info("Validating HRF temporal dynamics...")
        hrf_validation_results = validate_hrf_temporal_dynamics(
            times,
            hrf_hbo,
            epochs_fnirs,
            motor_roi_channel,
            onset_window=(
                config.analysis.hrf_onset_window_start_sec,
                config.analysis.hrf_onset_window_end_sec,
            ),
            peak_window=(
                config.analysis.hrf_peak_window_start_sec,
                config.analysis.hrf_peak_window_end_sec,
            ),
            plateau_window=(
                config.analysis.task_window_start_sec,
                config.analysis.task_window_end_sec,
            ),
        )
        onset_time = hrf_validation_results.get('onset_time_sec')
        time_to_peak = hrf_validation_results.get('time_to_peak_sec')
        logger.info(
            f"HRF onset: {onset_time if onset_time is not None else 'N/A'}s "
            f"(detected: {hrf_validation_results.get('onset_detected', False)})"
        )
        logger.info(
            f"Time-to-peak: {time_to_peak if time_to_peak is not None else 'N/A'}s "
            f"(plausible: {hrf_validation_results.get('peak_plausible', False)})"
        )

        # Compute HRF quality metrics
        logger.info("Computing HRF quality metrics...")
        quality_metrics = compute_hrf_quality_metrics(epochs_fnirs, motor_roi_channel)
        logger.info(
            f"Trial consistency: r={quality_metrics['trial_consistency_r']:.3f}"
        )
        logger.info(f"SNR: {quality_metrics['snr']:.2f}")

        # Create HRF validation dataclass
        hrf_validation = HRFValidation(
            channel=motor_roi_channel,
            onset_time_sec=hrf_validation_results.get("onset_time_sec", 0.0) or 0.0,
            onset_detected=hrf_validation_results.get("onset_detected", False),
            time_to_peak_sec=hrf_validation_results.get("time_to_peak_sec", 0.0) or 0.0,
            peak_plausible=hrf_validation_results.get("peak_within_range", False),
            plateau_amplitude_um=hrf_validation_results.get("plateau_amplitude", 0.0) or 0.0,
            plateau_p_value=hrf_validation_results.get("plateau_pvalue", 1.0) or 1.0,
            plateau_significant=hrf_validation_results.get("plateau_significant", False),
            trial_consistency_r=quality_metrics.get("trial_consistency_r", 0.0) or 0.0,
            consistency_acceptable=(quality_metrics.get("trial_consistency_r", 0.0) or 0.0) > 0.5,
        )

        # Generate fNIRS visualizations
        logger.info("Generating fNIRS visualizations...")
        hrf_curves = plot_hrf_curves(times, hrf_hbo, hrf_hbr, epochs_fnirs, motor_roi_channel)
        # Skip spatial map if no montage config available
        hrf_spatial = None
        try:
            hrf_spatial = plot_hrf_spatial_map(epochs_fnirs, fnirs_metadata.get("ChMontage", {}))
        except Exception as e:
            logger.warning(f"Could not generate HRF spatial map: {e}")

    except Exception as e:
        raise PipelineError(f"Stage 7 failed (fNIRS Analysis): {e}") from e


    # =========================================================================
    # STAGE 8: Multimodal coupling (Envelope → Cross-correlation)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 8: Neurovascular Coupling Analysis")
    logger.info("=" * 70)

    try:
        # Extract EEG alpha envelope from preprocessed raw data
        logger.info("Extracting EEG alpha envelope...")
        eeg_times, eeg_envelope = extract_eeg_envelope(
            raw_eeg_processed,
            motor_channel,
            freq_band=(config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz),
        )
        logger.info(f"Envelope extracted: {len(eeg_envelope)} samples")

        # Extract continuous HbO data from processed fNIRS for coupling analysis
        logger.info("Extracting continuous HbO data for coupling analysis...")
        hbo_channels = [ch for ch in raw_fnirs_processed.ch_names if "hbo" in ch.lower()]
        if not hbo_channels:
            raise ValueError("No HbO channels found in processed fNIRS data")
        
        # Use the motor ROI channel if available, otherwise first HbO channel
        coupling_channel = motor_roi_channel if motor_roi_channel in hbo_channels else hbo_channels[0]
        fnirs_hbo_continuous = raw_fnirs_processed.get_data(picks=[coupling_channel])[0]
        fnirs_times = raw_fnirs_processed.times
        logger.info(f"Using channel {coupling_channel} for coupling analysis ({len(fnirs_hbo_continuous)} samples)")

        # Compute neurovascular coupling
        logger.info("Computing neurovascular coupling...")
        coupling_results = compute_neurovascular_coupling(
            eeg_envelope, fnirs_hbo_continuous, eeg_times, fnirs_times, raw_fnirs_processed.info["sfreq"]
        )
        logger.info(
            f"Max correlation: {coupling_results['max_correlation']:.3f} "
            f"at lag {coupling_results['lag_seconds']:.2f}s"
        )
        logger.info(
            f"EEG precedes fNIRS: {coupling_results['lag_negative']}"
        )

        # Create coupling metrics dataclass
        coupling_metrics = CouplingMetrics(
            max_correlation=coupling_results["max_correlation"],
            lag_seconds=coupling_results["lag_seconds"],
            eeg_precedes_fnirs=coupling_results["lag_negative"],
            correlation_adequate=abs(coupling_results["max_correlation"]) > 0.4,
        )

        # Generate coupling visualization
        logger.info("Generating coupling visualization...")
        # Get resampled envelope for plotting
        eeg_envelope_resampled = resample_to_fnirs(
            eeg_envelope, eeg_times, times, epochs_fnirs.info["sfreq"]
        )
        coupling_overlay = plot_coupling_overlay(
            eeg_envelope_resampled,
            hrf_hbo,
            times,
            times,
            coupling_results,
            channel_eeg=motor_channel,
            channel_fnirs=motor_roi_channel,
        )

    except Exception as e:
        raise PipelineError(f"Stage 8 failed (Multimodal Coupling): {e}") from e


    # =========================================================================
    # STAGE 9: Generate reports (Quality, HTML, JSON)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 9: Generating Validation Reports")
    logger.info("=" * 70)

    try:
        # Generate quality report
        logger.info("Generating quality report...")
        tsv_path, json_path, png_path = generate_quality_report(
            quality_report,
            output_dir,
            subject_id,
            session_id,
            task,
            heatmap_figure=quality_heatmap,
        )
        logger.info(f"Quality TSV: {tsv_path.name}")
        logger.info(f"Quality JSON: {json_path.name}")
        if png_path:
            logger.info(f"Quality heatmap: {png_path.name}")

        # Create validation results dataclass
        validation_results = ValidationResults(
            subject_id=subject_id,
            session_id=session_id,
            task=task,
            timestamp=timestamp,
            software_versions=software_versions,
            config=config.to_dict(),
            quality_report=quality_report,
            erd_metrics=erd_metrics,
            erd_metrics_c4=erd_metrics_c4,
            hrf_validation=hrf_validation,
            coupling_metrics=coupling_metrics,
            lateralization_metrics=lateralization_metrics,
        )

        # Save numerical results
        logger.info("Saving numerical results...")
        metrics_path = save_numerical_results(
            validation_results, output_dir, subject_id, session_id, task
        )
        logger.info(f"Metrics JSON: {metrics_path.name}")

        # Generate HTML report
        logger.info("Generating HTML validation report...")
        figures = {
            "quality_heatmap": quality_heatmap,
            "eeg_spectrogram": eeg_spectrogram,
            "erd_timecourse": erd_timecourse,
            "eeg_spectrogram_left_by_condition": eeg_spectrogram_left_by_condition,
            "eeg_spectrogram_right_by_condition": eeg_spectrogram_right_by_condition,
            "hrf_curves": hrf_curves,
            "hrf_spatial": hrf_spatial,
            "coupling_overlay": coupling_overlay,
            "lateralization_timecourse": lateralization_timecourse_fig,
            "lateralization_barplot": lateralization_barplot_fig,
        }
        html_path = generate_validation_report_html(
            validation_results, figures, output_dir, subject_id, session_id, task
        )
        logger.info(f"HTML report: {html_path.name}")

    except Exception as e:
        raise PipelineError(f"Stage 9 failed (Report Generation): {e}") from e

    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info(f"Subject: sub-{subject_id}, Session: ses-{session_id}, Task: {task}")
    logger.info("=" * 70)

    return validation_results



def main() -> None:
    """
    Command-line interface for the validation pipeline.

    This function provides a CLI for running the validation pipeline on
    multimodal EEG + fNIRS data. It supports both required and optional
    arguments for flexible usage.

    Usage:
        python -m affective_fnirs.pipeline \\
            --xdf-file data/raw/sub-002/sub-002_recording.xdf \\
            --eeg-json data/raw/sub-002/sub-002_eeg.json \\
            --fnirs-json data/raw/sub-002/sub-002_fnirs.json \\
            --config configs/validation_pipeline.yml \\
            --output data/derivatives/validation-pipeline

    Requirements:
        - Req. 10.1: Accept configuration files
        - Validate input paths before execution
        - Support both CLI and library usage
    """
    import argparse
    import sys

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Multimodal EEG + fNIRS Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m affective_fnirs.pipeline \\
      --xdf-file data/raw/sub-002/sub-002_recording.xdf \\
      --eeg-json data/raw/sub-002/sub-002_eeg.json \\
      --fnirs-json data/raw/sub-002/sub-002_fnirs.json

  # Run with custom configuration
  python -m affective_fnirs.pipeline \\
      --xdf-file data/raw/sub-002/sub-002_recording.xdf \\
      --eeg-json data/raw/sub-002/sub-002_eeg.json \\
      --fnirs-json data/raw/sub-002/sub-002_fnirs.json \\
      --config configs/validation_pipeline.yml \\
      --output data/derivatives/validation-pipeline

  # Run with custom random seed
  python -m affective_fnirs.pipeline \\
      --xdf-file data/raw/sub-002/sub-002_recording.xdf \\
      --eeg-json data/raw/sub-002/sub-002_eeg.json \\
      --fnirs-json data/raw/sub-002/sub-002_fnirs.json \\
      --seed 12345

For more information, see the documentation at:
https://github.com/your-repo/affective-fnirs
        """,
    )

    # Required arguments
    parser.add_argument(
        "--xdf-file",
        type=Path,
        required=True,
        help="Path to XDF file containing EEG, fNIRS, and marker streams",
    )
    parser.add_argument(
        "--eeg-json",
        type=Path,
        required=True,
        help="Path to EEG JSON sidecar with channel metadata",
    )
    parser.add_argument(
        "--fnirs-json",
        type=Path,
        required=True,
        help="Path to fNIRS JSON sidecar with montage configuration",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (uses defaults if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for derivatives (default: data/derivatives/validation-pipeline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config if provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="affective-fnirs 0.1.0",
        help="Show version and exit",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate input paths before execution
    logger.info("Validating input paths...")
    if not args.xdf_file.exists():
        logger.error(f"XDF file not found: {args.xdf_file}")
        sys.exit(1)
    if not args.eeg_json.exists():
        logger.error(f"EEG JSON not found: {args.eeg_json}")
        sys.exit(1)
    if not args.fnirs_json.exists():
        logger.error(f"fNIRS JSON not found: {args.fnirs_json}")
        sys.exit(1)

    # Load configuration
    if args.config:
        if not args.config.exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        logger.info(f"Loading configuration from: {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = PipelineConfig.default()

    # Override random seed if provided
    if args.seed is not None:
        logger.info(f"Overriding random seed: {args.seed}")
        # Create new config with updated seed
        config_dict = config.to_dict()
        config_dict["random_seed"] = args.seed
        config = PipelineConfig.from_dict(config_dict)

    # Run pipeline
    try:
        logger.info("Starting validation pipeline...")
        results = run_validation_pipeline(
            xdf_file=args.xdf_file,
            eeg_json=args.eeg_json,
            fnirs_json=args.fnirs_json,
            config=config,
            output_dir=args.output,
        )

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Subject: sub-{results.subject_id}")
        logger.info(f"Session: ses-{results.session_id}")
        logger.info(f"Task: {results.task}")
        logger.info("")
        logger.info("Quality Assessment:")
        logger.info(
            f"  Bad channels: {results.quality_report.n_bad_channels}/{results.quality_report.n_total_channels}"
        )
        logger.info(f"  Mean SCI: {results.quality_report.mean_sci:.3f}")
        logger.info("")
        logger.info("EEG Analysis:")
        logger.info(
            f"  Alpha ERD: {results.erd_metrics.alpha_erd_percent:.1f}% "
            f"(p={results.erd_metrics.alpha_p_value:.4f}, "
            f"significant={results.erd_metrics.alpha_significant})"
        )
        logger.info(
            f"  Beta ERD: {results.erd_metrics.beta_erd_percent:.1f}% "
            f"(p={results.erd_metrics.beta_p_value:.4f}, "
            f"significant={results.erd_metrics.beta_significant})"
        )
        logger.info("")
        logger.info("fNIRS Analysis:")
        logger.info(
            f"  HRF onset: {results.hrf_validation.onset_time_sec:.2f}s "
            f"(detected={results.hrf_validation.onset_detected})"
        )
        logger.info(
            f"  Time-to-peak: {results.hrf_validation.time_to_peak_sec:.2f}s "
            f"(plausible={results.hrf_validation.peak_plausible})"
        )
        logger.info(
            f"  Plateau significant: {results.hrf_validation.plateau_significant}"
        )
        logger.info("")
        logger.info("Neurovascular Coupling:")
        logger.info(
            f"  Max correlation: {results.coupling_metrics.max_correlation:.3f}"
        )
        logger.info(f"  Lag: {results.coupling_metrics.lag_seconds:.2f}s")
        logger.info(
            f"  EEG precedes fNIRS: {results.coupling_metrics.eeg_precedes_fnirs}"
        )
        logger.info("=" * 70)

        logger.info("Pipeline completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
