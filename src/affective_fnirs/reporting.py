"""
Reporting Module for Multimodal Validation Pipeline.

This module generates comprehensive validation reports including:
- BIDS-compliant quality assessment tables (TSV + JSON)
- HTML validation reports with visualizations
- Numerical results for reproducibility
- Validation conclusions with pass/fail criteria

References:
    - BIDS specification: https://bids-specification.readthedocs.io
    - MNE Report: https://mne.tools/stable/generated/mne.Report.html
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChannelQuality:
    """Quality metrics for a single fNIRS channel."""

    channel_name: str
    sci: float
    saturation_percent: float
    cardiac_power: float
    cv: float
    is_bad: bool
    reason: str


@dataclass
class QualityReport:
    """Complete quality assessment report for fNIRS data."""

    channels: list[ChannelQuality]
    n_total_channels: int
    n_bad_channels: int
    mean_sci: float
    mean_saturation: float
    mean_cardiac_power: float
    mean_cv: float


@dataclass
class ERDMetrics:
    """Event-Related Desynchronization metrics for EEG."""

    channel: str
    alpha_erd_percent: float
    alpha_p_value: float
    alpha_significant: bool
    beta_erd_percent: float
    beta_p_value: float
    beta_significant: bool
    beta_rebound_percent: float
    beta_rebound_observed: bool


@dataclass
class HRFValidation:
    """Hemodynamic Response Function validation metrics."""

    channel: str
    onset_time_sec: float
    onset_detected: bool
    time_to_peak_sec: float
    peak_plausible: bool
    plateau_amplitude_um: float
    plateau_p_value: float
    plateau_significant: bool
    trial_consistency_r: float
    consistency_acceptable: bool


@dataclass
class CouplingMetrics:
    """Neurovascular coupling metrics."""

    max_correlation: float
    lag_seconds: float
    eeg_precedes_fnirs: bool
    correlation_adequate: bool


@dataclass
class LateralizationMetrics:
    """Lateralization analysis results for motor tasks."""

    # Condition-specific ERD values (alpha band)
    left_c3_alpha_erd: float
    left_c4_alpha_erd: float
    right_c3_alpha_erd: float
    right_c4_alpha_erd: float
    nothing_c3_alpha_erd: float
    nothing_c4_alpha_erd: float

    # Condition-specific ERD values (beta band)
    left_c3_beta_erd: float
    left_c4_beta_erd: float
    right_c3_beta_erd: float
    right_c4_beta_erd: float
    nothing_c3_beta_erd: float
    nothing_c4_beta_erd: float

    # Lateralization indices
    left_lateralization_alpha: float
    right_lateralization_alpha: float
    left_lateralization_beta: float
    right_lateralization_beta: float

    # Statistical tests
    left_vs_nothing_c4_p: float
    right_vs_nothing_c3_p: float
    left_contralateral_vs_ipsilateral_p: float
    right_contralateral_vs_ipsilateral_p: float

    # Validation flags
    left_shows_contralateral_erd: bool
    right_shows_contralateral_erd: bool
    lateralization_pattern_valid: bool

    # Trial counts
    n_left_trials: int
    n_right_trials: int
    n_nothing_trials: int


@dataclass
class EEGChannelQuality:
    """Quality metrics for a single EEG channel."""

    channel_name: str
    mean_correlation: float
    signal_variance: float
    amplitude_range_uv: float  # Peak-to-peak amplitude in µV
    std_uv: float  # Standard deviation in µV
    quality_status: str  # "good", "fair", "poor"
    is_bad: bool


@dataclass
class ExperimentQA:
    """Quality assurance metrics for experiment recording."""

    eeg_duration_sec: float
    fnirs_duration_sec: float
    eeg_n_valid_trials: int
    fnirs_n_valid_trials: int
    eeg_expected_trials: int
    fnirs_expected_trials: int
    eeg_duration_complete: bool
    fnirs_duration_complete: bool
    trials_match: bool
    eeg_channel_quality: list[EEGChannelQuality]


@dataclass
class ValidationResults:
    """Complete validation results for all analyses."""

    subject_id: str
    session_id: str
    task: str
    timestamp: str
    software_versions: dict[str, str]
    config: dict[str, Any]
    quality_report: QualityReport
    erd_metrics: ERDMetrics
    hrf_validation: HRFValidation
    coupling_metrics: CouplingMetrics
    experiment_qa: ExperimentQA
    lateralization_metrics: LateralizationMetrics | None = None
    erd_metrics_c4: ERDMetrics | None = None  # C4 ERD metrics


def generate_quality_report(
    quality_report: QualityReport,
    output_path: Path,
    subject_id: str,
    session_id: str,
    task: str,
    heatmap_figure: matplotlib.figure.Figure | None = None,
) -> tuple[Path, Path, Path | None]:
    """
    Generate fNIRS quality assessment report with BIDS-compliant outputs.

    Creates three output files following BIDS naming conventions:
    1. TSV file: Quality metrics table
    2. JSON file: Data dictionary describing TSV columns
    3. PNG file: Spatial heatmap of channel quality (optional)

    Args:
        quality_report: QualityReport dataclass with all metrics
        output_path: Output directory (data/derivatives/validation-pipeline/sub-{subject}/)
        subject_id: Subject identifier (e.g., '002')
        session_id: Session identifier (e.g., '001')
        task: Task name (e.g., 'fingertapping')
        heatmap_figure: Optional matplotlib Figure with spatial quality heatmap

    Returns:
        Tuple of (tsv_path, json_path, png_path)
        png_path is None if heatmap_figure not provided

    Requirements:
        - Req. 3.9: Generate quality report with all metrics
        - Req. 8.1: Include quality tables in validation report
        - Req. 9.2: Write to derivatives directory
        - Req. 9.3: Follow BIDS naming conventions
        - Req. 9.4: Create JSON data dictionaries

    Example:
        >>> tsv_path, json_path, png_path = generate_quality_report(
        ...     quality_report,
        ...     Path('data/derivatives/validation-pipeline/sub-002'),
        ...     subject_id='002',
        ...     session_id='001',
        ...     task='fingertapping',
        ...     heatmap_figure=fig
        ... )
    """
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate BIDS-compliant filenames
    base_filename = f"sub-{subject_id}_ses-{session_id}_task-{task}_desc-quality"
    tsv_filename = f"{base_filename}_channels.tsv"
    json_filename = f"{base_filename}_channels.json"
    png_filename = f"{base_filename}_heatmap.png"

    tsv_path = output_path / tsv_filename
    json_path = output_path / json_filename
    png_path = output_path / png_filename if heatmap_figure else None

    # Create DataFrame from quality report
    channel_data = []
    for ch_quality in quality_report.channels:
        channel_data.append(
            {
                "channel_name": ch_quality.channel_name,
                "sci": ch_quality.sci,
                "saturation_percent": ch_quality.saturation_percent,
                "cardiac_power": ch_quality.cardiac_power,
                "cv": ch_quality.cv,
                "is_bad": ch_quality.is_bad,
                "reason": ch_quality.reason,
            }
        )

    df = pd.DataFrame(channel_data)

    # Save TSV file (tab-separated, BIDS standard)
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.4f")

    # Create JSON data dictionary
    data_dictionary = {
        "channel_name": {
            "Description": "fNIRS channel identifier (source-detector pair with wavelength)",
            "Units": "n/a",
        },
        "sci": {
            "Description": "Scalp Coupling Index - correlation between wavelengths in cardiac band (Pollonini et al., 2016)",
            "Units": "correlation coefficient",
            "Range": "0-1",
            "Threshold": ">0.8 indicates good optode-scalp coupling",
            "Reference": "PMC4752525",
        },
        "saturation_percent": {
            "Description": "Percentage of samples exceeding 95% of ADC range",
            "Units": "%",
            "Range": "0-100",
            "Threshold": "<5% acceptable",
        },
        "cardiac_power": {
            "Description": "Peak Spectral Power in cardiac band (0.5-2.5 Hz) - PHOEBE framework",
            "Units": "normalized power",
            "Range": "0-1",
            "Threshold": ">0.1 indicates clear cardiac pulsation",
            "Reference": "PMC4752525",
        },
        "cv": {
            "Description": "Coefficient of Variation in baseline periods - (std/mean)*100",
            "Units": "%",
            "Range": "0-100",
            "Threshold": "<15% indicates stable signal",
            "Reference": "PMC7677693",
        },
        "is_bad": {
            "Description": "Channel marked as bad based on quality thresholds",
            "Units": "boolean",
        },
        "reason": {
            "Description": "Reason(s) for marking channel as bad (if applicable)",
            "Units": "text",
        },
    }

    # Save JSON data dictionary
    with open(json_path, "w") as f:
        json.dump(data_dictionary, f, indent=2)

    # Save heatmap if provided
    if heatmap_figure and png_path:
        heatmap_figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(heatmap_figure)

    return tsv_path, json_path, png_path


def create_data_dictionary(
    field_descriptions: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """
    Generate BIDS-compliant JSON data dictionary.

    Args:
        field_descriptions: Dictionary mapping field names to metadata
            Example: {
                'sci': {
                    'Description': '...',
                    'Units': '...',
                    'Range': '...',
                    'Threshold': '...'
                }
            }

    Returns:
        JSON-serializable dictionary for data dictionary

    Example:
        >>> descriptions = {
        ...     'sci': {
        ...         'Description': 'Scalp Coupling Index',
        ...         'Units': 'correlation coefficient',
        ...         'Range': '0-1',
        ...         'Threshold': '>0.8'
        ...     }
        ... }
        >>> data_dict = create_data_dictionary(descriptions)
    """
    return field_descriptions


def save_numerical_results(
    validation_results: ValidationResults,
    output_path: Path,
    subject_id: str,
    session_id: str,
    task: str,
) -> Path:
    """
    Save numerical validation results to JSON for reproducibility.

    Enables independent verification and reanalysis without re-running pipeline.

    Args:
        validation_results: Complete ValidationResults dataclass
        output_path: Output directory
        subject_id: Subject identifier
        session_id: Session identifier
        task: Task name

    Returns:
        Path to JSON file

    Requirements:
        - Req. 10.2: Log random seed for reproducibility
        - Req. 10.3: Save configuration with results

    Filename: sub-{subject}_ses-{session}_task-{task}_desc-validation_metrics.json

    Example:
        >>> json_path = save_numerical_results(
        ...     validation_results,
        ...     Path('data/derivatives/validation-pipeline/sub-002'),
        ...     subject_id='002',
        ...     session_id='001',
        ...     task='fingertapping'
        ... )
    """
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate BIDS-compliant filename
    filename = f"sub-{subject_id}_ses-{session_id}_task-{task}_desc-validation_metrics.json"
    json_path = output_path / filename

    # Convert dataclass to dictionary
    results_dict = asdict(validation_results)

    # Custom JSON encoder to handle numpy types
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_dict = convert_numpy_types(results_dict)

    # Save to JSON with pretty formatting
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    return json_path



def generate_validation_report_html(
    validation_results: ValidationResults,
    figures: dict[str, matplotlib.figure.Figure],
    output_path: Path,
    subject_id: str,
    session_id: str,
    task: str,
) -> Path:
    """
    Generate comprehensive HTML validation report using MNE Report.

    Creates a multi-section HTML report organized in 3 supra-sections:
    1. EEG Analysis: PSD, spectrograms, ERD metrics, lateralization
    2. fNIRS Analysis: Quality, HRF curves, temporal validation
    3. EEG + fNIRS: Multimodal coupling and integration

    Args:
        validation_results: Complete ValidationResults dataclass
        figures: Dictionary mapping figure names to matplotlib Figure objects
            Expected keys:
            - 'eeg_psd_by_condition': PSD for C3, C4, F3, F4 by condition
            - 'eeg_spectrogram': Time-frequency representation
            - 'erd_timecourse': ERD/ERS time course
            - 'quality_heatmap': fNIRS quality visualization
            - 'hrf_curves': HbO/HbR averaged curves
            - 'coupling_overlay': EEG envelope vs HbO overlay
        output_path: Output directory
        subject_id: Subject identifier
        session_id: Session identifier
        task: Task name

    Returns:
        Path to generated HTML report

    Requirements:
        - Req. 8.1: Include quality tables
        - Req. 8.2: Include EEG spectrograms
        - Req. 8.3: Include HRF curves
        - Req. 8.4: Include coupling plots
        - Req. 8.5: Output in HTML format
        - Req. 8.6: Include validation conclusions

    Filename: sub-{subject}_ses-{session}_task-{task}_desc-validation_report.html
    """
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate BIDS-compliant filename
    filename = f"sub-{subject_id}_ses-{session_id}_task-{task}_desc-validation_report.html"
    html_path = output_path / filename

    # Create MNE Report
    report = mne.Report(title=f"Validation Report: sub-{subject_id}")

    # =========================================================================
    # HEADER
    # =========================================================================
    header_html = f"""
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
        <h2>Multimodal Validation Pipeline Report</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td><strong>Subject:</strong></td>
                <td>sub-{subject_id}</td>
                <td><strong>Session:</strong></td>
                <td>ses-{session_id}</td>
            </tr>
            <tr>
                <td><strong>Task:</strong></td>
                <td>{task}</td>
                <td><strong>Timestamp:</strong></td>
                <td>{validation_results.timestamp}</td>
            </tr>
        </table>
        <h3>Software Versions</h3>
        <ul>
            {''.join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in validation_results.software_versions.items()])}
        </ul>
    </div>
    """
    report.add_html(header_html, title="Header")

    # =========================================================================
    # EXPERIMENT QA SECTION
    # =========================================================================
    qa_html = _generate_experiment_qa_section_html(validation_results.experiment_qa)
    report.add_html(qa_html, title="Experiment QA")

    # =========================================================================
    # SUPRA-SECTION 1: EEG ANALYSIS
    # =========================================================================
    eeg_section_html = """
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h2 style="color: #1976d2;">📊 SUPRA-SECTION 1: EEG ANALYSIS</h2>
        <p>Analysis of motor cortex activity during finger tapping task using 4 good EEG channels (C3, C4, F3, F4).</p>
    </div>
    """
    report.add_html(eeg_section_html, title="EEG Analysis")

    # 1.0: Power Spectral Density (Preprocessed Data)
    if "eeg_psd" in figures and figures["eeg_psd"] is not None:
        # Embed PSD image directly in HTML with full width control
        import base64
        from pathlib import Path
        
        # Handle both Path objects and strings
        psd_value = figures["eeg_psd"]
        if isinstance(psd_value, (str, Path)):
            psd_path = Path(psd_value)
            
            if psd_path.exists():
                with open(psd_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                psd_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0 EEG Power Spectral Density (Preprocessed)</h3>
                    <img src="data:image/png;base64,{img_data}" 
                         style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0 auto;" 
                         alt="EEG Power Spectral Density">
                </div>
                """
                report.add_html(psd_html, title="1.0 EEG Power Spectral Density (Preprocessed)")
            else:
                logger.warning(f"PSD plot file not found: {psd_path}")
        else:
            logger.warning(f"Unexpected type for eeg_psd: {type(psd_value)}")
    
    # 1.0.1: Left Hemisphere PSD by Condition
    if "eeg_psd_left" in figures and figures["eeg_psd_left"] is not None:
        import base64
        from pathlib import Path
        
        psd_left_value = figures["eeg_psd_left"]
        topo_left_value = figures.get("eeg_topo_left")
        
        if isinstance(psd_left_value, (str, Path)):
            psd_left_path = Path(psd_left_value)
            
            if psd_left_path.exists():
                with open(psd_left_path, "rb") as img_file:
                    psd_img_data = base64.b64encode(img_file.read()).decode()
                
                # Try to load topoplot if available
                topo_img_html = ""
                if topo_left_value and isinstance(topo_left_value, (str, Path)):
                    topo_left_path = Path(topo_left_value)
                    if topo_left_path.exists():
                        with open(topo_left_path, "rb") as img_file:
                            topo_img_data = base64.b64encode(img_file.read()).decode()
                        topo_img_html = f"""
                        <div style="flex: 0 0 300px; margin-left: 20px;">
                            <img src="data:image/png;base64,{topo_img_data}" 
                                 style="width: 100%; height: auto; display: block;" 
                                 alt="Left Hemisphere Topoplot">
                        </div>
                        """
                
                psd_left_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0.1 Left Hemisphere PSD by Condition</h3>
                    <p style="margin: 10px 0; color: #555; font-size: 14px;">
                        <strong>Electrodes:</strong> FC1, FC5, C3, CP1, CP5 (left motor cortex cluster)
                    </p>
                    <div style="display: flex; align-items: flex-start; gap: 20px;">
                        <div style="flex: 1;">
                            <img src="data:image/png;base64,{psd_img_data}" 
                                 style="width: 100%; max-width: 100%; height: auto; display: block;" 
                                 alt="Left Hemisphere PSD by Condition">
                        </div>
                        {topo_img_html}
                    </div>
                </div>
                """
                report.add_html(psd_left_html, title="1.0.1 Left Hemisphere PSD by Condition")
    
    # 1.0.2: Right Hemisphere PSD by Condition
    if "eeg_psd_right" in figures and figures["eeg_psd_right"] is not None:
        import base64
        from pathlib import Path
        
        psd_right_value = figures["eeg_psd_right"]
        topo_right_value = figures.get("eeg_topo_right")
        
        if isinstance(psd_right_value, (str, Path)):
            psd_right_path = Path(psd_right_value)
            
            if psd_right_path.exists():
                with open(psd_right_path, "rb") as img_file:
                    psd_img_data = base64.b64encode(img_file.read()).decode()
                
                # Try to load topoplot if available
                topo_img_html = ""
                if topo_right_value and isinstance(topo_right_value, (str, Path)):
                    topo_right_path = Path(topo_right_value)
                    if topo_right_path.exists():
                        with open(topo_right_path, "rb") as img_file:
                            topo_img_data = base64.b64encode(img_file.read()).decode()
                        topo_img_html = f"""
                        <div style="flex: 0 0 300px; margin-left: 20px;">
                            <img src="data:image/png;base64,{topo_img_data}" 
                                 style="width: 100%; height: auto; display: block;" 
                                 alt="Right Hemisphere Topoplot">
                        </div>
                        """
                
                psd_right_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0.2 Right Hemisphere PSD by Condition</h3>
                    <p style="margin: 10px 0; color: #555; font-size: 14px;">
                        <strong>Electrodes:</strong> FC2, FC6, C4, CP2, CP6 (right motor cortex cluster)
                    </p>
                    <div style="display: flex; align-items: flex-start; gap: 20px;">
                        <div style="flex: 1;">
                            <img src="data:image/png;base64,{psd_img_data}" 
                                 style="width: 100%; max-width: 100%; height: auto; display: block;" 
                                 alt="Right Hemisphere PSD by Condition">
                        </div>
                        {topo_img_html}
                    </div>
                </div>
                """
                report.add_html(psd_right_html, title="1.0.2 Right Hemisphere PSD by Condition")
    
    # 1.0.3: Time-Frequency Maps (TFR)
    if "eeg_tfr_maps" in figures and figures["eeg_tfr_maps"] is not None:
        import base64
        from pathlib import Path
        
        tfr_maps_value = figures["eeg_tfr_maps"]
        if isinstance(tfr_maps_value, (str, Path)):
            tfr_maps_path = Path(tfr_maps_value)
            
            if tfr_maps_path.exists():
                with open(tfr_maps_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                tfr_maps_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0.3 Time-Frequency Maps (Most Informative)</h3>
                    <p style="margin: 10px 0; color: #555; font-size: 14px;">
                        <strong>Time-Frequency Representation (TFR)</strong> showing power changes across time and frequency for motor cortex channels C3 and C4.
                        <br><br>
                        <strong>Expected patterns:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px; color: #555;">
                            <li><strong>Blue patch (ERD)</strong>: Power decrease in 8-30 Hz starting before movement onset (motor planning) and continuing during execution</li>
                            <li><strong>Red patch (ERS)</strong>: Power increase in Beta (~20 Hz) after movement ends (motor inhibition/rebound)</li>
                            <li><strong>Contralateral effect</strong>: C3 shows stronger ERD for RIGHT hand, C4 shows stronger ERD for LEFT hand</li>
                        </ul>
                        <br>
                        <strong>Color scale:</strong> Blue = desynchronization (ERD), Red = synchronization (ERS), White = no change from baseline
                    </p>
                    <img src="data:image/png;base64,{img_data}" 
                         style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0 auto;" 
                         alt="Time-Frequency Maps">
                </div>
                """
                report.add_html(tfr_maps_html, title="1.0.3 Time-Frequency Maps")
    
    # 1.0.4: Contralateral ERD/ERS Timecourse
    if "eeg_contralateral_timecourse" in figures and figures["eeg_contralateral_timecourse"] is not None:
        import base64
        from pathlib import Path
        
        contralateral_timecourse_value = figures["eeg_contralateral_timecourse"]
        if isinstance(contralateral_timecourse_value, (str, Path)):
            contralateral_timecourse_path = Path(contralateral_timecourse_value)
            
            if contralateral_timecourse_path.exists():
                with open(contralateral_timecourse_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                contralateral_timecourse_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0.4 Contralateral ERD/ERS Timecourse</h3>
                    <p style="margin: 10px 0; color: #555; font-size: 14px;">
                        Classic motor cortex desynchronization: C3 desynchronizes during RIGHT hand movement (contralateral), 
                        C4 desynchronizes during LEFT hand movement (contralateral). Shows alpha (8-13 Hz) and beta (13-30 Hz) bands.
                    </p>
                    <img src="data:image/png;base64,{img_data}" 
                         style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0 auto;" 
                         alt="Contralateral ERD/ERS Timecourse">
                </div>
                """
                report.add_html(contralateral_timecourse_html, title="1.0.4 Contralateral ERD/ERS Timecourse")
    
    # 1.0.5: Contralateral ERD/ERS Topoplots
    if "eeg_contralateral_topoplot" in figures and figures["eeg_contralateral_topoplot"] is not None:
        import base64
        from pathlib import Path
        
        contralateral_topoplot_value = figures["eeg_contralateral_topoplot"]
        if isinstance(contralateral_topoplot_value, (str, Path)):
            contralateral_topoplot_path = Path(contralateral_topoplot_value)
            
            if contralateral_topoplot_path.exists():
                with open(contralateral_topoplot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                contralateral_topoplot_html = f"""
                <div style="page-break-inside: avoid; margin: 20px 0;">
                    <h3>1.0.5 Contralateral ERD/ERS Spatial Distribution</h3>
                    <p style="margin: 10px 0; color: #555; font-size: 14px;">
                        Topographic maps showing ERD/ERS spatial distribution during task execution (2-4s after onset). 
                        Blue = desynchronization (ERD), Red = synchronization (ERS). 
                        Bottom row shows LEFT-RIGHT contrast highlighting contralateral effect.
                    </p>
                    <img src="data:image/png;base64,{img_data}" 
                         style="width: 100%; max-width: 100%; height: auto; display: block; margin: 0 auto;" 
                         alt="Contralateral ERD/ERS Topoplots">
                </div>
                """
                report.add_html(contralateral_topoplot_html, title="1.0.5 Contralateral ERD/ERS Topoplots")

    # 1.1: Power Spectral Density by Condition
    if "eeg_psd_by_condition" in figures and figures["eeg_psd_by_condition"] is not None:
        report.add_figure(
            figures["eeg_psd_by_condition"],
            title="1.1 Power Spectral Density by Condition",
            caption="PSD for C3, C4, F3, F4 across LEFT, RIGHT, and NOTHING conditions. "
            "Solid lines show mean power, shaded areas show ±1 SD across trials (n=7 per condition). "
            "Gray bands indicate alpha (8-13 Hz) and beta (13-30 Hz) frequency ranges.",
        )

    # 1.2: ERD/ERS Metrics
    erd_html = _generate_erd_section_html(validation_results.erd_metrics)
    report.add_html(erd_html, title="1.2 ERD/ERS Metrics")

    # 1.3: Condition Contrast Spectrograms
    if "eeg_spectrogram" in figures:
        report.add_figure(
            figures["eeg_spectrogram"],
            title="1.3 Condition Contrast Spectrograms",
            caption="Lateralization specificity revealed through condition contrasts. "
            "Top: C3 showing LEFT - RIGHT. Bottom: C4 showing RIGHT - LEFT. "
            "Blue regions indicate expected contralateral ERD patterns.",
        )

    # 1.4: ERD/ERS Time Course
    if "erd_timecourse" in figures:
        report.add_figure(
            figures["erd_timecourse"],
            title="1.4 ERD/ERS Time Course (Bilateral)",
            caption="Alpha and beta band power changes over time for C3 and C4. "
            "Negative values = ERD (power decrease), positive values = ERS (power increase).",
        )

    # 1.5: Spectrograms by Condition
    if "eeg_spectrogram_left_by_condition" in figures and figures["eeg_spectrogram_left_by_condition"] is not None:
        report.add_figure(
            figures["eeg_spectrogram_left_by_condition"],
            title="1.5 Left Motor Cortex (C3) - By Condition",
            caption="Time-frequency spectrograms for LEFT, RIGHT, and NOTHING conditions in C3. "
            "Expected: RIGHT hand movement → ERD in C3 (contralateral control).",
        )

    if "eeg_spectrogram_right_by_condition" in figures and figures["eeg_spectrogram_right_by_condition"] is not None:
        report.add_figure(
            figures["eeg_spectrogram_right_by_condition"],
            title="1.6 Right Motor Cortex (C4) - By Condition",
            caption="Time-frequency spectrograms for LEFT, RIGHT, and NOTHING conditions in C4. "
            "Expected: LEFT hand movement → ERD in C4 (contralateral control).",
        )

    # 1.7: Lateralization Analysis
    if validation_results.lateralization_metrics is not None:
        lat_html = _generate_lateralization_section_html(validation_results.lateralization_metrics)
        report.add_html(lat_html, title="1.7 Lateralization Analysis")

        if "lateralization_timecourse" in figures and figures["lateralization_timecourse"] is not None:
            report.add_figure(
                figures["lateralization_timecourse"],
                title="1.8 Lateralization Time-Course",
                caption="ERD/ERS time-course for LEFT, RIGHT, and NOTHING conditions. "
                "Expected: LEFT hand → C4 ERD, RIGHT hand → C3 ERD (contralateral control).",
            )

        if "lateralization_barplot" in figures and figures["lateralization_barplot"] is not None:
            report.add_figure(
                figures["lateralization_barplot"],
                title="1.9 Lateralization ERD Comparison",
                caption="Bar plot comparing ERD across conditions and hemispheres.",
            )

    # =========================================================================
    # SUPRA-SECTION 2: fNIRS ANALYSIS
    # =========================================================================
    fnirs_section_html = """
    <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h2 style="color: #f57c00;">🧠 SUPRA-SECTION 2: fNIRS ANALYSIS</h2>
        <p>Analysis of hemodynamic response in motor cortex during finger tapping task.</p>
    </div>
    """
    report.add_html(fnirs_section_html, title="fNIRS Analysis")

    # 2.1: Quality Assessment
    quality_html = _generate_quality_section_html(validation_results.quality_report)
    report.add_html(quality_html, title="2.1 Quality Assessment")

    if "quality_heatmap" in figures:
        report.add_figure(
            figures["quality_heatmap"],
            title="2.2 Quality Heatmap",
            caption="Spatial distribution of fNIRS channel quality. Green = good channels (SCI > 0.8), red = bad channels.",
        )

    # 2.3: HRF Analysis
    hrf_html = _generate_hrf_section_html(validation_results.hrf_validation)
    report.add_html(hrf_html, title="2.3 Hemodynamic Response Function")

    if "hrf_curves" in figures:
        report.add_figure(
            figures["hrf_curves"],
            title="2.4 HRF Curves",
            caption=f"HbO/HbR concentration changes. "
            f"Onset: {validation_results.hrf_validation.onset_time_sec:.1f}s, "
            f"Peak: {validation_results.hrf_validation.time_to_peak_sec:.1f}s, "
            f"Amplitude: {validation_results.hrf_validation.plateau_amplitude_um:.2f} μM",
        )

    # =========================================================================
    # SUPRA-SECTION 3: EEG + fNIRS (MULTIMODAL)
    # =========================================================================
    multimodal_section_html = """
    <div style="background-color: #f3e5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h2 style="color: #7b1fa2;">🔗 SUPRA-SECTION 3: EEG + fNIRS (MULTIMODAL COUPLING)</h2>
        <p>Analysis of neurovascular coupling between EEG and fNIRS signals.</p>
    </div>
    """
    report.add_html(multimodal_section_html, title="EEG + fNIRS")

    # 3.1: Coupling Metrics
    coupling_html = _generate_coupling_section_html(validation_results.coupling_metrics)
    report.add_html(coupling_html, title="3.1 Neurovascular Coupling Metrics")

    if "coupling_overlay" in figures:
        report.add_figure(
            figures["coupling_overlay"],
            title="3.2 Coupling Overlay",
            caption=f"EEG alpha envelope vs HbO concentration. "
            f"Lag: {validation_results.coupling_metrics.lag_seconds:.1f}s, "
            f"Correlation: r={validation_results.coupling_metrics.max_correlation:.2f}",
        )

    # =========================================================================
    # VALIDATION SUMMARY
    # =========================================================================
    conclusions = generate_validation_conclusions(validation_results)
    report.add_html(
        f'<div style="background-color: #e8f4f8; padding: 20px; border-radius: 5px;"><pre>{conclusions}</pre></div>',
        title="Validation Summary",
    )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    config_html = _generate_config_section_html(validation_results.config)
    report.add_html(config_html, title="Configuration")

    # Save report
    report.save(html_path, overwrite=True, open_browser=False)

    return html_path


def compute_eeg_channel_quality(
    raw: mne.io.Raw, 
    channels: list[str],
    known_good_channels: list[str] | None = None
) -> list[EEGChannelQuality]:
    """
    Compute quality metrics for RAW EEG channels (before preprocessing).

    Evaluates channel quality based on state-of-the-art criteria for raw EEG:
    1. Signal amplitude range (5-100µV for raw data)
    2. Kurtosis (detect muscle artifacts and flat signals)
    3. Spectral power in physiological bands (alpha/beta)
    4. Correlation with other channels (spatial consistency)

    Quality classification:
    - Good: Normal amplitude (10-100µV), physiological spectrum, good correlation
    - Fair: Borderline amplitude or moderate artifacts
    - Poor: Very low amplitude (<5µV), flat signal, or excessive noise

    Args:
        raw: MNE Raw object with UNPROCESSED EEG data (before filtering/re-referencing)
        channels: List of channel names to evaluate (e.g., ['C3', 'C4', 'F3', 'F4'])
        known_good_channels: Optional list of channels known to be well-connected
            for correlation-based quality assessment

    Returns:
        List of EEGChannelQuality dataclasses with metrics for each channel

    Notes:
        - MUST be called on RAW data before preprocessing (Grosselin et al., 2019)
        - Amplitude thresholds (5-100µV) are for raw data, not preprocessed
        - After CAR/filtering, variance drops to e-13/e-14 range (normal)
        - Kurtosis > 5 indicates muscle artifacts or spiky noise
        - Correlation > 0.6-0.7 is normal for clean EEG (Grosselin et al., 2019)

    References:
        - Grosselin et al. (2019). Quality assessment of single-channel EEG for wearable devices.
          MDPI Sensors 19(3):601. DOI: 10.3390/s19030601

    Example:
        >>> # Call BEFORE preprocessing
        >>> quality = compute_eeg_channel_quality(raw_eeg, all_channels)
        >>> # Then preprocess
        >>> processed_eeg = preprocess_eeg_pipeline(raw_eeg, config)
    """
    # Get EEG channel data
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]
    data_eeg = raw.get_data(picks=eeg_picks)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_eeg)
    np.fill_diagonal(corr_matrix, np.nan)
    mean_corr_all = np.nanmean(corr_matrix, axis=1)

    # Compute amplitude metrics for each channel
    variance_all = np.var(data_eeg, axis=1)
    std_all = np.std(data_eeg, axis=1)
    range_all = np.ptp(data_eeg, axis=1)  # Peak-to-peak amplitude
    
    # Compute kurtosis (detect artifacts and flat signals)
    # Kurtosis > 5 indicates spiky artifacts (muscle, eye blinks)
    # Kurtosis < 1 indicates flat/constant signal
    from scipy import stats as scipy_stats
    kurtosis_all = scipy_stats.kurtosis(data_eeg, axis=1, fisher=True)

    # If known good channels provided, compute correlation with them
    if known_good_channels:
        known_good_indices = [
            eeg_channel_names.index(ch) 
            for ch in known_good_channels 
            if ch in eeg_channel_names
        ]
    else:
        known_good_indices = None

    # Evaluate specified channels
    channel_qualities = []
    for ch_name in channels:
        if ch_name not in eeg_channel_names:
            # Channel not found - mark as poor
            channel_qualities.append(
                EEGChannelQuality(
                    channel_name=ch_name,
                    mean_correlation=0.0,
                    signal_variance=0.0,
                    amplitude_range_uv=0.0,
                    std_uv=0.0,
                    quality_status="poor",
                    is_bad=True,
                )
            )
            continue

        # Get channel index and metrics
        ch_idx = eeg_channel_names.index(ch_name)
        mean_corr = mean_corr_all[ch_idx]
        variance = variance_all[ch_idx]
        std_dev = std_all[ch_idx]
        amplitude_range = range_all[ch_idx]
        kurtosis = kurtosis_all[ch_idx]

        # Convert to µV for amplitude thresholds (raw data scale)
        std_dev_uv = std_dev * 1e6
        amplitude_range_uv = amplitude_range * 1e6

        # Compute correlation with known good channels (if provided)
        if known_good_indices and ch_name not in known_good_channels:
            corr_with_good = corr_matrix[ch_idx, known_good_indices]
            mean_corr_with_good = np.nanmean(corr_with_good)
        else:
            mean_corr_with_good = None

        # Quality assessment based on state-of-the-art criteria (Grosselin et al., 2019)
        # NOTE: Using std instead of range as primary metric because range is sensitive to
        # outliers and DC offset. Typical raw EEG std: 10-50 µV (Grosselin et al., 2019)
        
        # CRITERION 1: Signal variability (std) - primary for disconnection detection
        # Raw EEG: std > 5 µV is normal, std < 2 µV suggests disconnection/flat signal
        if std_dev_uv < 0.5:
            # Very low std - definitely disconnected or flat signal
            quality_status = "poor"
            is_bad = True
        elif std_dev_uv > 100.0:
            # Excessive std - likely excessive artifacts or poor contact
            quality_status = "poor"
            is_bad = True
        elif std_dev_uv < 2.0:
            # Low std - questionable quality (possible poor contact)
            quality_status = "fair"
            is_bad = False
        else:
            # Normal std (2-100 µV) - check additional criteria
            
            # CRITERION 2: Kurtosis (artifact detection)
            if kurtosis > 10.0:
                # Very high kurtosis - excessive spiky artifacts
                quality_status = "poor"
                is_bad = True
            elif kurtosis < 0.5:
                # Very low kurtosis - possibly flat/constant signal
                quality_status = "poor"
                is_bad = True
            elif kurtosis > 5.0:
                # Moderate kurtosis - some artifacts present
                quality_status = "fair"
                is_bad = False
            else:
                # Normal kurtosis - check correlation
                
                # CRITERION 3: Spatial correlation
                if mean_corr_with_good is not None:
                    # Use correlation with known good channels
                    if mean_corr_with_good < 0.3:
                        # Low correlation - likely noise/artifacts
                        quality_status = "poor"
                        is_bad = True
                    elif mean_corr_with_good < 0.5:
                        # Moderate correlation - questionable
                        quality_status = "fair"
                        is_bad = False
                    else:
                        # Good correlation (>0.5) - channel is good
                        quality_status = "good"
                        is_bad = False
                else:
                    # No ground truth - use general correlation
                    # Grosselin et al.: correlation > 0.6-0.7 is normal
                    if mean_corr > 0.6:
                        quality_status = "good"
                        is_bad = False
                    elif mean_corr > 0.3:
                        quality_status = "fair"
                        is_bad = False
                    else:
                        # Low correlation - questionable
                        quality_status = "poor"
                        is_bad = True

        channel_qualities.append(
            EEGChannelQuality(
                channel_name=ch_name,
                mean_correlation=float(mean_corr),
                signal_variance=float(variance),
                amplitude_range_uv=float(amplitude_range_uv),
                std_uv=float(std_dev_uv),
                quality_status=quality_status,
                is_bad=is_bad,
            )
        )

    return channel_qualities


def _generate_experiment_qa_section_html(experiment_qa: ExperimentQA) -> str:
    """Generate HTML for experiment QA section."""
    eeg_duration_status = "✓" if experiment_qa.eeg_duration_complete else "✗"
    fnirs_duration_status = "✓" if experiment_qa.fnirs_duration_complete else "✗"
    trials_match_status = "✓" if experiment_qa.trials_match else "✗"

    html = f"""
    <h3>Experiment Quality Assurance</h3>
    <p>Recording duration and trial count validation for EEG and fNIRS modalities.</p>
    
    <h4>Recording Duration</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Modality</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Duration (s)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Status</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">EEG</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.eeg_duration_sec:.1f}s</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {'green' if experiment_qa.eeg_duration_complete else 'red'};">{eeg_duration_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">fNIRS</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.fnirs_duration_sec:.1f}s</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {'green' if experiment_qa.fnirs_duration_complete else 'red'};">{fnirs_duration_status}</td>
        </tr>
    </table>
    
    <h4>Valid Trials</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Modality</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Valid Trials</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Expected Trials</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Percentage</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">EEG</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.eeg_n_valid_trials}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.eeg_expected_trials}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {'green' if experiment_qa.eeg_n_valid_trials >= experiment_qa.eeg_expected_trials else 'red'};">{(experiment_qa.eeg_n_valid_trials / experiment_qa.eeg_expected_trials * 100):.1f}%</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">fNIRS</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.fnirs_n_valid_trials}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{experiment_qa.fnirs_expected_trials}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {'green' if experiment_qa.fnirs_n_valid_trials >= experiment_qa.fnirs_expected_trials else 'red'};">{(experiment_qa.fnirs_n_valid_trials / experiment_qa.fnirs_expected_trials * 100):.1f}%</td>
        </tr>
    </table>
    
    <h4>EEG Channel Quality (Raw Data Metrics)</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Channel</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Amplitude Range (µV)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Std Dev (µV)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Mean Correlation</th>
        </tr>
    """

    # Add rows for each channel
    for ch_quality in experiment_qa.eeg_channel_quality:
        html += f"""
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">{ch_quality.channel_name}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{ch_quality.amplitude_range_uv:.2f}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{ch_quality.std_uv:.2f}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{ch_quality.mean_correlation:.3f}</td>
        </tr>
        """

    html += """
    </table>
    """

    return html


def _generate_quality_section_html(quality_report: QualityReport) -> str:
    """Generate HTML for quality assessment section."""
    html = f"""
    <h3>fNIRS Quality Metrics Summary</h3>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Total Channels</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.n_total_channels}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Bad Channels</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.n_bad_channels}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Mean SCI</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.mean_sci:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Mean Saturation %</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.mean_saturation:.2f}%</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Mean Cardiac Power</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.mean_cardiac_power:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Mean CV %</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{quality_report.mean_cv:.2f}%</td>
        </tr>
    </table>
    
    <h3>Bad Channels Details</h3>
    """

    if quality_report.n_bad_channels > 0:
        html += '<table style="border-collapse: collapse; width: 100%;">'
        html += """
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Channel</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Reason</th>
        </tr>
        """
        for ch in quality_report.channels:
            if ch.is_bad:
                html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">{ch.channel_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{ch.reason}</td>
                </tr>
                """
        html += "</table>"
    else:
        html += '<p style="color: green; font-weight: bold;">✓ All channels passed quality assessment</p>'

    return html


def _generate_erd_section_html(erd_metrics: ERDMetrics) -> str:
    """Generate HTML for EEG ERD/ERS section."""
    alpha_status = "✓" if erd_metrics.alpha_significant else "✗"
    beta_status = "✓" if erd_metrics.beta_significant else "✗"
    rebound_status = "✓" if erd_metrics.beta_rebound_observed else "✗"

    html = f"""
    <h3>Event-Related Desynchronization (ERD) Analysis</h3>
    <p><strong>Analysis Channel:</strong> {erd_metrics.channel}</p>
    
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Status</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Alpha ERD (8-13 Hz)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{erd_metrics.alpha_erd_percent:.1f}% (p={erd_metrics.alpha_p_value:.4f})</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{alpha_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Beta ERD (13-30 Hz)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{erd_metrics.beta_erd_percent:.1f}% (p={erd_metrics.beta_p_value:.4f})</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{beta_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Beta Rebound (post-task)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{erd_metrics.beta_rebound_percent:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{rebound_status}</td>
        </tr>
    </table>
    
    <p><strong>Expected Patterns:</strong></p>
    <ul>
        <li>Mu ERD: -20% to -40% during task (motor cortex activation)</li>
        <li>Beta ERD: -30% to -50% during movement</li>
        <li>Beta rebound: +10% to +30% after task cessation</li>
    </ul>
    """

    return html


def _generate_hrf_section_html(hrf_validation: HRFValidation) -> str:
    """Generate HTML for fNIRS HRF section."""
    onset_status = "✓" if hrf_validation.onset_detected else "✗"
    peak_status = "✓" if hrf_validation.peak_plausible else "✗"
    amplitude_status = "✓" if hrf_validation.plateau_significant else "✗"
    consistency_status = "✓" if hrf_validation.consistency_acceptable else "✗"

    html = f"""
    <h3>Hemodynamic Response Function (HRF) Validation</h3>
    <p><strong>Analysis Channel:</strong> {hrf_validation.channel}</p>
    
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Status</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">HbO Onset Time</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{hrf_validation.onset_time_sec:.1f}s</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{onset_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Time-to-Peak</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{hrf_validation.time_to_peak_sec:.1f}s</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{peak_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Plateau Amplitude</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{hrf_validation.plateau_amplitude_um:.2f} μM (p={hrf_validation.plateau_p_value:.4f})</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{amplitude_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Trial Consistency</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">r={hrf_validation.trial_consistency_r:.2f}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{consistency_status}</td>
        </tr>
    </table>
    
    <p><strong>Expected Patterns:</strong></p>
    <ul>
        <li>HbO onset: 2-3s post-stimulus (neurovascular delay)</li>
        <li>Time-to-peak: 4-8s for brief stimuli</li>
        <li>Plateau amplitude: Significantly positive (p < 0.05)</li>
        <li>Trial consistency: r > 0.5 indicates reliable response</li>
    </ul>
    """

    return html


def _generate_coupling_section_html(coupling_metrics: CouplingMetrics) -> str:
    """Generate HTML for neurovascular coupling section."""
    lag_status = "✓" if coupling_metrics.eeg_precedes_fnirs else "✗"
    corr_status = "✓" if coupling_metrics.correlation_adequate else "✗"

    html = f"""
    <h3>Neurovascular Coupling Analysis</h3>
    
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Status</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Maximum Correlation</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">r={coupling_metrics.max_correlation:.2f}</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{corr_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Optimal Lag</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{coupling_metrics.lag_seconds:.1f}s</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{lag_status}</td>
        </tr>
    </table>
    
    <p><strong>Expected Patterns:</strong></p>
    <ul>
        <li>Negative lag: EEG changes precede HbO changes by 2-5s</li>
        <li>Correlation strength: r > 0.4 indicates adequate coupling</li>
        <li>Interpretation: Neural activity drives hemodynamic response</li>
    </ul>
    """

    return html


def _generate_lateralization_section_html(lat_metrics: LateralizationMetrics) -> str:
    """Generate HTML for lateralization analysis section."""
    left_status = "✓" if lat_metrics.left_shows_contralateral_erd else "✗"
    right_status = "✓" if lat_metrics.right_shows_contralateral_erd else "✗"
    overall_status = "✓" if lat_metrics.lateralization_pattern_valid else "✗"

    html = f"""
    <h3>Motor Lateralization Analysis</h3>
    <p>Analysis of contralateral ERD patterns for LEFT vs RIGHT hand movement conditions.</p>
    
    <h4>Trial Counts</h4>
    <table style="border-collapse: collapse; width: 50%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Trials</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.n_left_trials}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.n_right_trials}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">NOTHING</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.n_nothing_trials}</td>
        </tr>
    </table>
    
    <h4>Alpha Band ERD (8-13 Hz)</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">C3 (left)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">C4 (right)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Lateralization</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.left_c3_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold; color: {'green' if lat_metrics.left_c4_alpha_erd < -5 else 'red'};">{lat_metrics.left_c4_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.left_lateralization_alpha:.1f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold; color: {'green' if lat_metrics.right_c3_alpha_erd < -5 else 'red'};">{lat_metrics.right_c3_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.right_c4_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.right_lateralization_alpha:.1f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">NOTHING</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.nothing_c3_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.nothing_c4_alpha_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">-</td>
        </tr>
    </table>
    
    <h4>Beta Band ERD (13-30 Hz)</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px;">Condition</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">C3 (left)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">C4 (right)</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Lateralization</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.left_c3_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold; color: {'green' if lat_metrics.left_c4_beta_erd < -5 else 'red'};">{lat_metrics.left_c4_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.left_lateralization_beta:.1f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; font-weight: bold; color: {'green' if lat_metrics.right_c3_beta_erd < -5 else 'red'};">{lat_metrics.right_c3_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.right_c4_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.right_lateralization_beta:.1f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">NOTHING</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.nothing_c3_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.nothing_c4_beta_erd:.1f}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">-</td>
        </tr>
    </table>
    
    <h4>Statistical Tests</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px;">Test</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">p-value</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT C4 vs NOTHING C4</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {'green' if lat_metrics.left_vs_nothing_c4_p < 0.05 else 'red'};">{lat_metrics.left_vs_nothing_c4_p:.4f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT C3 vs NOTHING C3</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right; color: {'green' if lat_metrics.right_vs_nothing_c3_p < 0.05 else 'red'};">{lat_metrics.right_vs_nothing_c3_p:.4f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT: C4 vs C3 (paired)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.left_contralateral_vs_ipsilateral_p:.4f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT: C3 vs C4 (paired)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{lat_metrics.right_contralateral_vs_ipsilateral_p:.4f}</td>
        </tr>
    </table>
    
    <h4>Validation Summary</h4>
    <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
        <tr style="background-color: #f0f0f0;">
            <th style="border: 1px solid #ddd; padding: 8px;">Criterion</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Status</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">LEFT shows contralateral ERD (C4)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {'green' if lat_metrics.left_shows_contralateral_erd else 'red'};">{left_status}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">RIGHT shows contralateral ERD (C3)</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {'green' if lat_metrics.right_shows_contralateral_erd else 'red'};">{right_status}</td>
        </tr>
        <tr style="background-color: {'#d4edda' if lat_metrics.lateralization_pattern_valid else '#f8d7da'};">
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Overall Lateralization Pattern Valid</td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;">{overall_status}</td>
        </tr>
    </table>
    
    <p><strong>Expected Patterns:</strong></p>
    <ul>
        <li>LEFT hand movement → ERD in C4 (right hemisphere, contralateral)</li>
        <li>RIGHT hand movement → ERD in C3 (left hemisphere, contralateral)</li>
        <li>NOTHING condition → No significant ERD (baseline control)</li>
        <li>ERD threshold: < -5% indicates significant desynchronization</li>
    </ul>
    """

    return html


def _generate_config_section_html(config: dict[str, Any]) -> str:
    """Generate HTML for configuration section."""
    html = """
    <h3>Pipeline Configuration</h3>
    <p>Configuration parameters used for this analysis:</p>
    <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
    """
    html += json.dumps(config, indent=2)
    html += """
    </pre>
    """
    return html


def generate_validation_conclusions(validation_results: ValidationResults) -> str:
    """
    Generate text conclusions based on validation criteria.

    Evaluates pass/fail for each criterion and provides overall assessment.

    Args:
        validation_results: Complete ValidationResults dataclass

    Returns:
        Formatted text string with conclusions

    Requirements:
        - Req. 8.6: Include validation conclusions

    Criteria:
        EEG:
        - Alpha ERD significant? (p < 0.05, magnitude > 20%)
        - Beta ERD significant? (p < 0.05, magnitude > 30%)
        - Beta rebound observed? (positive change post-task)

        fNIRS:
        - HbO onset detected? (within 2-3s)
        - Time-to-peak plausible? (within 4-8s)
        - Plateau amplitude significant? (p < 0.05)
        - Trial consistency acceptable? (r > 0.5)

        Coupling:
        - EEG precedes fNIRS? (negative lag)
        - Correlation strength adequate? (r > 0.4)

    Example:
        >>> conclusions = generate_validation_conclusions(validation_results)
        >>> print(conclusions)
    """
    erd = validation_results.erd_metrics
    hrf = validation_results.hrf_validation
    coupling = validation_results.coupling_metrics

    # Evaluate EEG criteria
    eeg_criteria = []
    eeg_criteria.append(
        (
            "Alpha ERD significant",
            erd.alpha_significant and erd.alpha_erd_percent < -20,
            f"{erd.alpha_erd_percent:.1f}% (p={erd.alpha_p_value:.4f})",
        )
    )
    eeg_criteria.append(
        (
            "Beta ERD significant",
            erd.beta_significant and erd.beta_erd_percent < -30,
            f"{erd.beta_erd_percent:.1f}% (p={erd.beta_p_value:.4f})",
        )
    )
    eeg_criteria.append(
        (
            "Beta rebound observed",
            erd.beta_rebound_observed and erd.beta_rebound_percent > 0,
            f"{erd.beta_rebound_percent:.1f}% post-task",
        )
    )

    # Evaluate fNIRS criteria
    fnirs_criteria = []
    fnirs_criteria.append(
        (
            "HbO onset detected",
            hrf.onset_detected,
            f"{hrf.onset_time_sec:.1f}s (expected: 2-3s)",
        )
    )
    fnirs_criteria.append(
        (
            "Time-to-peak plausible",
            hrf.peak_plausible,
            f"{hrf.time_to_peak_sec:.1f}s (expected: 4-8s)",
        )
    )
    fnirs_criteria.append(
        (
            "Plateau amplitude significant",
            hrf.plateau_significant,
            f"{hrf.plateau_amplitude_um:.2f} μM (p={hrf.plateau_p_value:.4f})",
        )
    )
    fnirs_criteria.append(
        (
            "Trial consistency acceptable",
            hrf.consistency_acceptable,
            f"r={hrf.trial_consistency_r:.2f} (threshold: >0.5)",
        )
    )

    # Evaluate coupling criteria
    coupling_criteria = []
    coupling_criteria.append(
        (
            "EEG precedes fNIRS",
            coupling.eeg_precedes_fnirs,
            f"lag={coupling.lag_seconds:.1f}s (negative expected)",
        )
    )
    coupling_criteria.append(
        (
            "Correlation strength adequate",
            coupling.correlation_adequate,
            f"r={coupling.max_correlation:.2f} (threshold: >0.4)",
        )
    )

    # Generate conclusions text
    conclusions = "=" * 70 + "\n"
    conclusions += "VALIDATION CRITERIA ASSESSMENT\n"
    conclusions += "=" * 70 + "\n\n"

    # EEG section
    conclusions += "EEG Analysis (Motor Cortex Activation):\n"
    conclusions += "-" * 70 + "\n"
    for name, passed, value in eeg_criteria:
        status = "[PASS]" if passed else "[FAIL]"
        conclusions += f"{status} {name}: {value}\n"

    eeg_passed = sum(1 for _, passed, _ in eeg_criteria if passed)
    conclusions += f"\nEEG Summary: {eeg_passed}/{len(eeg_criteria)} criteria met\n\n"

    # fNIRS section
    conclusions += "fNIRS Analysis (Hemodynamic Response):\n"
    conclusions += "-" * 70 + "\n"
    for name, passed, value in fnirs_criteria:
        status = "[PASS]" if passed else "[FAIL]"
        conclusions += f"{status} {name}: {value}\n"

    fnirs_passed = sum(1 for _, passed, _ in fnirs_criteria if passed)
    conclusions += f"\nfNIRS Summary: {fnirs_passed}/{len(fnirs_criteria)} criteria met\n\n"

    # Coupling section
    conclusions += "Neurovascular Coupling:\n"
    conclusions += "-" * 70 + "\n"
    for name, passed, value in coupling_criteria:
        status = "[PASS]" if passed else "[FAIL]"
        conclusions += f"{status} {name}: {value}\n"

    coupling_passed = sum(1 for _, passed, _ in coupling_criteria if passed)
    conclusions += f"\nCoupling Summary: {coupling_passed}/{len(coupling_criteria)} criteria met\n\n"

    # Overall assessment
    total_criteria = len(eeg_criteria) + len(fnirs_criteria) + len(coupling_criteria)
    total_passed = eeg_passed + fnirs_passed + coupling_passed
    pass_rate = (total_passed / total_criteria) * 100

    conclusions += "=" * 70 + "\n"
    conclusions += f"OVERALL: {total_passed}/{total_criteria} criteria met ({pass_rate:.0f}%)\n"
    conclusions += "=" * 70 + "\n\n"

    # Recommendation
    if pass_rate >= 80:
        conclusions += "✓ RECOMMENDATION: Paradigm validated. All major criteria met.\n"
        conclusions += "  The pilot data demonstrates:\n"
        conclusions += "  - Clear motor cortex activation (ERD patterns)\n"
        conclusions += "  - Robust hemodynamic response (HbO increase)\n"
        conclusions += "  - Strong neurovascular coupling (EEG leads fNIRS)\n\n"
        conclusions += "  Proceed with full study implementation.\n"
    elif pass_rate >= 60:
        conclusions += "⚠ RECOMMENDATION: Partial validation. Some criteria not met.\n"
        conclusions += "  Review failed criteria and consider:\n"
        conclusions += "  - Adjusting task parameters\n"
        conclusions += "  - Improving electrode/optode placement\n"
        conclusions += "  - Collecting additional pilot data\n"
    else:
        conclusions += "✗ RECOMMENDATION: Validation failed. Major issues detected.\n"
        conclusions += "  Significant problems with:\n"

        if eeg_passed < len(eeg_criteria) / 2:
            conclusions += "  - EEG signal quality or task engagement\n"
        if fnirs_passed < len(fnirs_criteria) / 2:
            conclusions += "  - fNIRS optode coupling or hemodynamic response\n"
        if coupling_passed < len(coupling_criteria) / 2:
            conclusions += "  - Neurovascular coupling strength\n"

        conclusions += "\n  Recommend troubleshooting before proceeding.\n"

    # Diagnostic suggestions for failures
    if pass_rate < 100:
        conclusions += "\nDiagnostic Suggestions:\n"
        conclusions += "-" * 70 + "\n"

        if not erd.alpha_significant:
            conclusions += "• Alpha ERD not significant: Check C3 electrode placement,\n"
            conclusions += "  verify subject engaged in task, consider longer task duration\n"

        if not erd.beta_significant:
            conclusions += "• Beta ERD not significant: Verify movement execution,\n"
            conclusions += "  check for excessive muscle artifacts\n"

        if not hrf.onset_detected:
            conclusions += "• HbO onset not detected: Check optode coupling (SCI),\n"
            conclusions += "  verify task timing, consider motion artifacts\n"

        if not hrf.plateau_significant:
            conclusions += "• Plateau amplitude not significant: Check optode placement,\n"
            conclusions += "  verify task difficulty sufficient to elicit response\n"

        if not coupling.eeg_precedes_fnirs:
            conclusions += "• Unexpected coupling lag: Verify temporal synchronization,\n"
            conclusions += "  check LSL timestamps, review event markers\n"

        if not coupling.correlation_adequate:
            conclusions += "• Weak coupling correlation: May indicate poor signal quality\n"
            conclusions += "  in either EEG or fNIRS, or weak task-related activation\n"

    return conclusions
