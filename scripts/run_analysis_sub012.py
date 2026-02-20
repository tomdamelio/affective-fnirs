#!/usr/bin/env python3
"""
Dedicated Analysis Script for Sub-012 (Modified Fingertapping Protocol).

This script implements the specific requirements for sub-012:
1. Handles 'sub-12' XDF filename discrepancy (vs 'sub-012')
2. Syncs NOTHING epochs from post-trial rest periods (10s task + 8s rest)
3. Runs standard validation pipeline (TFR, ERP, CSP)
4. Uses 'Agg' backend to prevent interactive plots from blocking execution.
5. Supports --fnirs-only and --eeg-only flags for single-modality runs.
"""
import matplotlib
# Force non-interactive backend BEFORE importing run_analysis or mne
matplotlib.use('Agg')

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional
import numpy as np
import mne

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import (
    load_xdf_file,
    identify_streams,
    extract_stream_data,
)
import json
from affective_fnirs.mne_builder import (
    build_eeg_raw,
    build_fnirs_raw,
    embed_events,
)
from affective_fnirs.validation import validate_nothing_condition
from affective_fnirs.fnirs_quality import calculate_sci
from affective_fnirs.fnirs_processing import (
    convert_to_optical_density,
    correct_motion_artifacts,
    identify_short_channels,
    apply_short_channel_regression,
    convert_to_hemoglobin,
    filter_hemoglobin_data,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import run_analysis as main_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ModalityMode(Enum):
    """Pipeline execution mode based on CLI flags.

    Attributes:
        FULL_MULTIMODAL: Default mode - process both EEG and fNIRS data.
        FNIRS_ONLY: Process only fNIRS data, skip EEG processing.
        EEG_ONLY: Process only EEG data, skip fNIRS processing.
    """

    FULL_MULTIMODAL = auto()  # Default: both EEG and fNIRS
    FNIRS_ONLY = auto()  # --fnirs-only flag
    EEG_ONLY = auto()  # --eeg-only flag


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with single-modality flags.

    Creates an ArgumentParser with --config option and mutually exclusive
    --fnirs-only and --eeg-only flags. The flags allow running the pipeline
    for a single modality without processing the other.

    Returns:
        Configured ArgumentParser with --fnirs-only and --eeg-only flags.
    """
    parser = argparse.ArgumentParser(
        description="Sub-012 Analysis Pipeline with single-modality support"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sub-012.yml"),
        help="Path to config file",
    )

    modality_group = parser.add_mutually_exclusive_group()
    modality_group.add_argument(
        "--fnirs-only",
        action="store_true",
        help="Run fNIRS pipeline only, skip EEG processing",
    )
    modality_group.add_argument(
        "--eeg-only",
        action="store_true",
        help="Run EEG pipeline only, skip fNIRS processing",
    )

    return parser


def determine_modality_mode(args: argparse.Namespace) -> ModalityMode:
    """Determine the modality mode from parsed CLI arguments.

    Args:
        args: Parsed command-line arguments containing fnirs_only and eeg_only flags.

    Returns:
        ModalityMode enum value indicating which modalities to process.
    """
    if args.fnirs_only:
        return ModalityMode.FNIRS_ONLY
    elif args.eeg_only:
        return ModalityMode.EEG_ONLY
    else:
        return ModalityMode.FULL_MULTIMODAL


# =============================================================================
# ROI Definitions for Sub-012 Montage
# =============================================================================

ROI_DEFINITIONS: dict[str, list[str]] = {
    "Left Anterior": ["S1_", "S9_"],
    "Right Anterior": ["S2_", "S10_"],
    "Left Posterior": ["S3_", "S4_", "S5_"],
    "Right Posterior": ["S6_", "S7_", "S8_"],
}
"""Mapping of ROI names to source prefixes for the sub-012 fNIRS montage.

Each ROI groups channels by their source label prefix:
- Left Anterior: Sources S1, S9 (frontal left hemisphere)
- Right Anterior: Sources S2, S10 (frontal right hemisphere)
- Left Posterior: Sources S3, S4, S5 (parietal/occipital left hemisphere)
- Right Posterior: Sources S6, S7, S8 (parietal/occipital right hemisphere)
"""

SHORT_CHANNEL_ROI_MAP: dict[str, dict[str, Any]] = {
    "S13_D1": {
        "roi": "Left Anterior",
        "long_sources": ["S1", "S9"],
    },
    "S14_D3": {
        "roi": "Right Anterior",
        "long_sources": ["S2", "S10"],
    },
    "S15_D6": {
        "roi": "Left Posterior",
        "long_sources": ["S3", "S4", "S5"],
    },
    "S16_D9": {
        "roi": "Right Posterior",
        "long_sources": ["S6", "S7", "S8"],
    },
}
"""Mapping of short channels to their ROI and associated long channel sources.

Each short channel (S13-S16) is paired with a specific detector and regresses
the superficial physiological noise from its spatially nearest long channels:
- S13_D1: Left Anterior region (S1, S9 cluster)
- S14_D3: Right Anterior region (S2, S10 cluster)
- S15_D6: Left Posterior region (S3, S4, S5 cluster)
- S16_D9: Right Posterior region (S6, S7, S8 cluster)

Used by SCR (Short Channel Regression) to verify proximity-based pairing
matches the expected anatomical mapping from the sub-012 montage.
"""


@dataclass(frozen=True)
class ROIDefinition:
    """Definition of a region of interest for fNIRS analysis.

    Immutable dataclass representing an anatomical ROI with its associated
    source prefixes and optional short channel for regression.

    Attributes:
        name: Human-readable ROI name (e.g., "Left Anterior").
        source_prefixes: Tuple of source label prefixes that belong to this ROI
            (e.g., ("S1_", "S9_")). The trailing underscore ensures exact prefix
            matching.
        short_channel: Optional short channel pair for this ROI's regression
            (e.g., "S13_D1"). None if no short channel is assigned.
    """

    name: str
    source_prefixes: tuple[str, ...]
    short_channel: Optional[str]

    def matches_channel(self, channel_name: str) -> bool:
        """Check if a channel belongs to this ROI based on source prefix.

        Extracts the source label from the channel name and checks if it
        matches any of the ROI's source prefixes.

        Args:
            channel_name: fNIRS channel name in format "S1_D1 760" or "S1_D1 hbo".

        Returns:
            True if the channel's source matches one of this ROI's prefixes.

        Example:
            >>> roi = ROIDefinition("Left Anterior", ("S1_", "S9_"), "S13_D1")
            >>> roi.matches_channel("S1_D2 760")
            True
            >>> roi.matches_channel("S3_D4 hbo")
            False
        """
        base_pair = channel_name.split(" ")[0]
        source_with_underscore = base_pair.split("_")[0] + "_"
        return source_with_underscore in self.source_prefixes


@dataclass
class ExternalBaselineResult:
    """Result of loading external baseline from a separate recording.

    Attributes:
        baseline_means: Dictionary mapping channel names to their mean baseline values.
        duration_sec: Duration of baseline segment used (in seconds).
        n_channels: Number of fNIRS channels with baseline values.
        source_file: Path to the baseline XDF file.
    """

    baseline_means: dict[str, float]
    duration_sec: float
    n_channels: int
    source_file: Path


def load_external_fnirs_baseline(
    baseline_xdf_path: Path,
    baseline_duration_sec: float = 60.0,
    marker_name: str = "baseline",
) -> ExternalBaselineResult:
    """Load fNIRS baseline from a separate baseline recording file.

    Extracts fNIRS intensity data from a dedicated baseline recording and
    computes the mean value per channel. This provides a more stable baseline
    than using a short pre-stimulus window from the main recording.

    Scientific rationale:
    - A 60-second baseline captures slow physiological drifts and provides
      a more representative "resting state" reference than 1-second windows.
    - External baseline recordings are acquired before the task, ensuring
      no contamination from task-related hemodynamic responses.

    Processing:
    1. Load XDF file and identify fNIRS stream
    2. Find baseline marker (if present) or use first 60s
    3. Extract baseline_duration_sec of data after marker
    4. Compute mean intensity per channel

    Args:
        baseline_xdf_path: Path to the baseline XDF file.
        baseline_duration_sec: Duration of baseline to extract (default: 60s).
        marker_name: Name of the baseline marker annotation (default: "baseline").

    Returns:
        ExternalBaselineResult with per-channel mean baseline values.

    Raises:
        FileNotFoundError: If baseline XDF file does not exist.
        ValueError: If no fNIRS stream found in baseline file.

    Example:
        >>> result = load_external_fnirs_baseline(
        ...     Path("data/raw/sub-012/ses-001/sub-12_ses-S001_task-baseline_run-001_eeg.xdf"),
        ...     baseline_duration_sec=60.0
        ... )
        >>> print(f"Loaded baseline for {result.n_channels} channels")
    """
    logger.info(f"Loading external fNIRS baseline from: {baseline_xdf_path}")

    if not baseline_xdf_path.exists():
        raise FileNotFoundError(f"Baseline XDF file not found: {baseline_xdf_path}")

    # Load XDF file
    streams, header = load_xdf_file(baseline_xdf_path)
    streams = identify_streams(streams)

    # Get fNIRS stream
    fnirs_stream = streams.get("fnirs")
    if fnirs_stream is None:
        raise ValueError(f"No fNIRS stream found in baseline file: {baseline_xdf_path}")

    # Extract fNIRS data
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
    
    # Ensure data is in (channels, samples) format
    # extract_stream_data should return (channels, samples), but verify
    n_channels_expected = int(fnirs_stream.get("info", {}).get("channel_count", ["0"])[0])
    if n_channels_expected > 0 and fnirs_data.shape[0] != n_channels_expected:
        # Data might be transposed (samples, channels) - fix it
        if fnirs_data.shape[1] == n_channels_expected:
            logger.info(f"Transposing baseline data from {fnirs_data.shape} to (channels, samples)")
            fnirs_data = fnirs_data.T
    
    n_channels = fnirs_data.shape[0]
    n_samples = fnirs_data.shape[1]
    logger.info(
        f"Baseline fNIRS: {n_channels} channels, "
        f"{n_samples} samples, {fnirs_sfreq:.2f} Hz"
    )

    # Find baseline marker onset (if marker stream exists)
    marker_stream = streams.get("markers")
    baseline_start_sample = 0

    if marker_stream is not None:
        marker_data, _, marker_timestamps = extract_stream_data(marker_stream)
        # Search for baseline marker
        marker_data_flat = marker_data.flatten()
        for idx in range(len(marker_data_flat)):
            marker_str = str(marker_data_flat[idx]).lower()
            if marker_name.lower() in marker_str:
                # Find corresponding sample in fNIRS data
                marker_time = marker_timestamps[idx]
                # Find closest fNIRS sample
                baseline_start_sample = int(
                    np.argmin(np.abs(fnirs_timestamps - marker_time))
                )
                logger.info(
                    f"Found '{marker_name}' marker at sample {baseline_start_sample} "
                    f"(time: {marker_time:.2f}s)"
                )
                break
        else:
            logger.warning(
                f"No '{marker_name}' marker found, using first {baseline_duration_sec}s"
            )
            baseline_start_sample = 0
    else:
        logger.warning(
            f"No marker stream in baseline file, using first {baseline_duration_sec}s"
        )

    # Calculate end sample - ensure we don't exceed data bounds
    n_samples_baseline = int(baseline_duration_sec * fnirs_sfreq)
    baseline_end_sample = min(baseline_start_sample + n_samples_baseline, n_samples)
    
    # Ensure start < end
    if baseline_start_sample >= baseline_end_sample:
        logger.warning(
            f"Invalid baseline range: start={baseline_start_sample}, end={baseline_end_sample}. "
            f"Using first {baseline_duration_sec}s instead."
        )
        baseline_start_sample = 0
        baseline_end_sample = min(n_samples_baseline, n_samples)

    actual_duration = (baseline_end_sample - baseline_start_sample) / fnirs_sfreq
    logger.info(
        f"Extracting baseline: samples {baseline_start_sample} to {baseline_end_sample} "
        f"({actual_duration:.1f}s)"
    )

    # Extract baseline segment and compute mean per channel
    baseline_segment = fnirs_data[:, baseline_start_sample:baseline_end_sample]
    baseline_means = {}

    # Get channel names from stream info - handle nested XML structure
    # XDF stream info can have various structures depending on the recording software
    ch_names = []
    try:
        stream_info = fnirs_stream.get("info", {}) if fnirs_stream else {}
        if stream_info is None:
            stream_info = {}
        
        desc = stream_info.get("desc", None)
        if desc is not None and isinstance(desc, list) and len(desc) > 0:
            desc_dict = desc[0] if desc[0] is not None else {}
            channels_container = desc_dict.get("channels", None) if isinstance(desc_dict, dict) else None
            if channels_container is not None and isinstance(channels_container, list) and len(channels_container) > 0:
                channels_dict = channels_container[0] if channels_container[0] is not None else {}
                channel_list = channels_dict.get("channel", []) if isinstance(channels_dict, dict) else []
                if channel_list:
                    for ch in channel_list:
                        if ch is not None and isinstance(ch, dict):
                            label = ch.get("label", [])
                            if label and isinstance(label, list) and len(label) > 0:
                                ch_names.append(str(label[0]))
        
        logger.info(f"Extracted {len(ch_names)} channel names from stream info")
    except Exception as exc:
        logger.warning(f"Could not extract channel names from stream info: {exc}")
        ch_names = []

    # Generate baseline means with channel names (or generic names if extraction failed)
    for ch_idx in range(n_channels):
        # Try to get channel name from extracted list
        if ch_idx < len(ch_names):
            ch_name = ch_names[ch_idx]
        else:
            ch_name = f"CH{ch_idx:03d}"

        baseline_means[ch_name] = float(np.mean(baseline_segment[ch_idx, :]))

    logger.info(
        f"Computed baseline means for {len(baseline_means)} channels "
        f"(duration: {actual_duration:.1f}s)"
    )

    return ExternalBaselineResult(
        baseline_means=baseline_means,
        duration_sec=actual_duration,
        n_channels=len(baseline_means),
        source_file=baseline_xdf_path,
    )


def apply_external_baseline_to_fnirs(
    raw_fnirs: mne.io.Raw,
    baseline_result: ExternalBaselineResult,
) -> mne.io.Raw:
    """Apply external baseline normalization to fNIRS intensity data.

    Divides intensity data by the baseline mean (I / I₀), which is the
    state-of-the-art approach for fNIRS baseline correction. This should
    be applied BEFORE optical density conversion.

    Scientific rationale (Beer-Lambert law):
    - OD = -log10(I / I₀) where I₀ is baseline intensity
    - By dividing by baseline mean, we normalize each channel's intensity
      relative to its resting-state level
    - This accounts for inter-channel differences in optical coupling,
      skin pigmentation, and hair interference
    - A 60-second external baseline provides stable I₀ estimation,
      superior to short pre-stimulus windows

    References:
    - Scholkmann et al. (2014) NeuroImage: "A review on continuous wave
      functional near-infrared spectroscopy"
    - Tachtsidis & Scholkmann (2016): fNIRS signal quality and artifacts

    Note:
    - Channels with baseline ≤ 0 are skipped (invalid for division)
    - If baseline channels have generic names (CH000, CH001...), matching
      is done by index position (assumes same channel order in both files)

    Args:
        raw_fnirs: MNE Raw object with fNIRS intensity data.
        baseline_result: ExternalBaselineResult from load_external_fnirs_baseline().

    Returns:
        MNE Raw object with baseline-normalized intensity data (I / I₀).
    """
    logger.info("Applying external baseline normalization (I / I₀) to fNIRS intensity...")

    # Work on a copy to avoid modifying original
    raw_corrected = raw_fnirs.copy()
    data = raw_corrected.get_data()

    n_corrected = 0
    n_missing = 0
    n_invalid = 0

    # Check if baseline has generic channel names (CH000, CH001, etc.)
    # If so, use index-based matching instead of name-based
    baseline_ch_names = list(baseline_result.baseline_means.keys())
    use_index_matching = all(
        ch_name.startswith("CH") and ch_name[2:].isdigit()
        for ch_name in baseline_ch_names
    )

    if use_index_matching:
        logger.info(
            "Baseline has generic channel names - using index-based matching"
        )
        # Convert baseline_means to list ordered by index
        baseline_values_by_idx = [
            baseline_result.baseline_means.get(f"CH{idx:03d}")
            for idx in range(len(baseline_ch_names))
        ]

    for ch_idx, ch_name in enumerate(raw_corrected.ch_names):
        baseline_value = None

        if use_index_matching:
            # Match by index position (same channel order in both recordings)
            if ch_idx < len(baseline_values_by_idx):
                baseline_value = baseline_values_by_idx[ch_idx]
        else:
            # Name-based matching
            # Extract base channel name (without wavelength suffix)
            # fNIRS channels in MNE are like "S1_D1 760" or "S1_D1 850"
            base_name = ch_name.split(" ")[0]

            # Direct match
            if ch_name in baseline_result.baseline_means:
                baseline_value = baseline_result.baseline_means[ch_name]
            # Try base name match (without wavelength)
            elif base_name in baseline_result.baseline_means:
                baseline_value = baseline_result.baseline_means[base_name]
            else:
                # Try fuzzy matching for channel naming differences
                for bl_ch_name, bl_value in baseline_result.baseline_means.items():
                    if base_name in bl_ch_name or bl_ch_name in base_name:
                        baseline_value = bl_value
                        break

        if baseline_value is not None:
            # Validate baseline value (must be positive for division)
            if baseline_value > 0:
                # Divide by baseline mean: I_normalized = I / I₀
                data[ch_idx, :] = data[ch_idx, :] / baseline_value
                n_corrected += 1
            else:
                n_invalid += 1
                if n_invalid <= 3:
                    logger.warning(
                        f"Invalid baseline value for {ch_name}: {baseline_value:.4f} "
                        "(must be > 0 for division)"
                    )
        else:
            n_missing += 1
            if n_missing <= 5:
                logger.warning(f"No baseline found for channel: {ch_name}")

    # Update raw data
    raw_corrected._data = data

    logger.info(
        f"External baseline normalization applied: {n_corrected} channels corrected, "
        f"{n_missing} missing, {n_invalid} invalid baseline values"
    )

    return raw_corrected


@dataclass
class SynthesisStats:
    """Tracks NOTHING annotation synthesis outcomes.

    Attributes:
        n_created: Number of NOTHING annotations successfully created.
        n_skipped: Number of trials skipped due to insufficient rest.
        n_source_trials: Total LEFT/RIGHT trials found as synthesis sources.
    """

    n_created: int
    n_skipped: int
    n_source_trials: int


def synthesize_nothing_annotations(
    raw: mne.io.Raw,
    task_duration: float = 7.0,
    rest_duration_cap: float = 6.0,
) -> tuple[mne.io.Raw, dict[str, int]]:
    """Synthesize NOTHING annotations from post-trial rest periods.

    Creates a NOTHING annotation for each LEFT/RIGHT trial whose subsequent
    rest period is long enough.  The virtual onset is placed 1.0 s after
    stimulus offset (task_end + 1.0) to allow a 1-s baseline gap, and the
    annotation duration equals ``rest_duration_cap`` (6.0 s by default),
    matching LEFT/RIGHT epoch length.

    Args:
        raw: MNE Raw object containing LEFT/RIGHT annotations.
        task_duration: Duration of the motor task in seconds.
        rest_duration_cap: Duration assigned to each NOTHING annotation
            (epoch length).  The minimum required rest between trials is
            ``rest_duration_cap + 1.0`` (baseline gap + epoch).

    Returns:
        A tuple of (modified raw, synthesis_stats) where synthesis_stats is
        a dict with keys ``n_created``, ``n_skipped``, ``n_source_trials``.
    """
    logger.info("Synthesizing NOTHING annotations from post-trial rest...")

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description
    sort_idx = np.argsort(onsets)
    sorted_onsets = onsets[sort_idx]
    sorted_descriptions = descriptions[sort_idx]

    new_onsets: list[float] = []
    new_durations: list[float] = []
    new_descriptions: list[str] = []

    n_source_trials = 0
    n_skipped = 0

    for trial_idx in range(len(sorted_onsets)):
        description = sorted_descriptions[trial_idx]
        onset = sorted_onsets[trial_idx]

        is_left = description == "LEFT" or description.startswith("LEFT/")
        is_right = description == "RIGHT" or description.startswith("RIGHT/")

        if not (is_left or is_right):
            continue

        n_source_trials += 1
        task_end = onset + task_duration
        nothing_virtual_onset = task_end + 1.0

        # Find next LEFT/RIGHT trial onset to bound available rest
        next_trial_onset: Optional[float] = None
        for search_idx in range(trial_idx + 1, len(sorted_onsets)):
            next_desc = sorted_descriptions[search_idx]
            if (
                next_desc == "LEFT"
                or next_desc.startswith("LEFT/")
                or next_desc == "RIGHT"
                or next_desc.startswith("RIGHT/")
            ):
                next_trial_onset = sorted_onsets[search_idx]
                break

        if next_trial_onset is not None:
            available_rest_duration = next_trial_onset - task_end
            if available_rest_duration >= (rest_duration_cap + 1.0):
                duration = rest_duration_cap
            else:
                n_skipped += 1
                continue
        else:
            # Last trial — assume sufficient rest
            duration = rest_duration_cap

        new_onsets.append(nothing_virtual_onset)
        new_durations.append(duration)
        new_descriptions.append("NOTHING")

    n_created = len(new_onsets)
    synthesis_stats: dict[str, int] = {
        "n_created": n_created,
        "n_skipped": n_skipped,
        "n_source_trials": n_source_trials,
    }

    if not new_onsets:
        logger.warning(
            "No NOTHING annotations created "
            f"(source trials: {n_source_trials}, skipped: {n_skipped})"
        )
        return raw, synthesis_stats

    nothing_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + nothing_annotations)
    logger.info(
        f"NOTHING synthesis complete — "
        f"created: {n_created}, skipped: {n_skipped}, "
        f"source trials: {n_source_trials}"
    )
    return raw, synthesis_stats


def generate_sci_comparison_plot(
    raw: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate grouped barplot of SCI values comparing first and second halves of recording.

    Args:
        raw: MNE Raw object (fNIRS intensity).
        output_path: Directory to save the plot.
        config: Subject configuration.

    Returns:
        Path to saved plot or None if failed.
    """
    logger.info("Generating SCI comparison plot (Initial vs Final)...")
    try:
        # Split data into two halves
        duration = raw.times[-1]
        midpoint = duration / 2.0
        
        raw_initial = raw.copy().crop(tmin=0, tmax=midpoint)
        # raw.crop is in-place and fails if tmin < raw.first_time, but raw is usually 0-anchored here
        # For the second half, we need a fresh copy of the original
        raw_final = raw.copy().crop(tmin=midpoint, tmax=None)
        
        # Calculate SCI for both halves
        # Note: calculate_sci expects freq_range=(0.5, 2.5) by default, 
        # which matches typical cardiac band.
        sci_initial = calculate_sci(raw_initial, sci_threshold=0.0) # threshold 0 to get all values
        sci_final = calculate_sci(raw_final, sci_threshold=0.0)
        
        # Prepare data for plotting
        data = []
        # sci_initial is dict {channel_pair: value}
        # Channel pairs are like 'S1_D1'
        
        all_pairs = sorted(list(set(list(sci_initial.keys()) + list(sci_final.keys()))))
        
        for pair in all_pairs:
            # Initial
            if pair in sci_initial:
                data.append({
                    'Channel Pair': pair,
                    'Segment': 'Initial (First Half)',
                    'SCI': sci_initial[pair]
                })
            
            # Final
            if pair in sci_final:
                data.append({
                    'Channel Pair': pair,
                    'Segment': 'Final (Second Half)',
                    'SCI': sci_final[pair]
                })
                
        if not data:
            logger.warning("No SCI data available for comparison plot")
            return None
            
        df = pd.DataFrame(data)
        
        # Plotting
        plt.figure(figsize=(15, 8))
        sns.set_theme(style="whitegrid")
        
        # Determine appropriate bar width and layout based on number of channels
        # If many channels, might need to rotate labels more or split plot
        
        ax = sns.barplot(
            data=df,
            x='Channel Pair',
            y='SCI',
            hue='Segment',
            palette=['#3498db', '#e74c3c'], # Blue for initial, Red for final
            alpha=0.8
        )
        
        # Add threshold line (typical good quality threshold)
        threshold = config.quality.sci_threshold
        plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
        
        plt.title(f'Scalp Coupling Index (SCI) Stability: Initial vs Final Segment\nSubject {config.subject.id}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Channel Pair', fontsize=12)
        plt.ylabel('Scalp Coupling Index (0-1)', fontsize=12)
        plt.ylim(0, 1.1) # SCI is 0-1, leave room for legend/text
        plt.legend(title='Recording Segment', loc='upper right')
        
        # Rotate x-axis labels if many channels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-sci_comparison.png"
        )
        filepath = output_path / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"SCI comparison plot saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate SCI comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_fnirs_timeseries_plot(
    raw_haemo: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
    good_channels: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Generate interactive full-experiment HbO/HbR time series plot in Z-scores.

    Creates an interactive HTML plot using Plotly with:
    - HbO (red) and HbR (blue) traces on the same plot (Z-scored)
    - Y-axis fixed to [-3, 3] for standardized visualization
    - Horizontal range slider for navigating through time
    - Zoom and pan capabilities (including arrow keys)
    - Stimulus markers color-coded by condition

    Args:
        raw_haemo: Preprocessed hemoglobin data (hbo/hbr channels).
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_channels: List of good channel pairs (e.g., ["S1_D1"]).
                       If None, uses all channels.

    Returns:
        Path to saved HTML file, or None if no good channels.
    """
    import plotly.graph_objects as go
    from scipy.stats import zscore

    logger.info("Generating interactive fNIRS time series plot (Z-scores)...")
    try:
        # Get data
        data = raw_haemo.get_data()  # (n_channels, n_times)
        times = raw_haemo.times
        ch_names = raw_haemo.ch_names

        # Filter channels to include only good_channels
        if good_channels is not None:
            hbo_indices = []
            hbr_indices = []

            for idx, ch_name in enumerate(ch_names):
                base_pair = ch_name.split(" ")[0]

                if base_pair in good_channels:
                    if "hbo" in ch_name:
                        hbo_indices.append(idx)
                    elif "hbr" in ch_name:
                        hbr_indices.append(idx)
        else:
            hbo_indices = [idx for idx, name in enumerate(ch_names) if "hbo" in name]
            hbr_indices = [idx for idx, name in enumerate(ch_names) if "hbr" in name]

        # Handle edge case: zero good channels
        if not hbo_indices:
            logger.warning("Zero good HbO channels available for time series plot")
            return None
        if not hbr_indices:
            logger.warning("Zero good HbR channels available for time series plot")
            return None

        logger.info(
            f"Computing Z-scored mean ± std across {len(hbo_indices)} HbO and "
            f"{len(hbr_indices)} HbR good channels"
        )

        # Z-score each channel individually, then compute mean
        hbo_zscored = np.array([zscore(data[idx, :]) for idx in hbo_indices])
        hbr_zscored = np.array([zscore(data[idx, :]) for idx in hbr_indices])

        # Calculate mean and std of Z-scored data
        hbo_mean = np.mean(hbo_zscored, axis=0)
        hbr_mean = np.mean(hbr_zscored, axis=0)
        hbo_std = np.std(hbo_zscored, axis=0)
        hbr_std = np.std(hbr_zscored, axis=0)

        # Create interactive figure with Plotly
        fig = go.Figure()

        # HbO shaded std band using a single filled area trace
        times_list = times.tolist()
        hbo_upper = (hbo_mean + hbo_std).tolist()
        hbo_lower = (hbo_mean - hbo_std).tolist()
        
        fig.add_trace(go.Scatter(
            x=times_list + times_list[::-1],
            y=hbo_upper + hbo_lower[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='HbO std band',
        ))

        # HbO mean trace
        fig.add_trace(go.Scatter(
            x=times,
            y=hbo_mean,
            mode='lines',
            name=f'HbO (n={len(hbo_indices)} ch)',
            line=dict(color='red', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>HbO: %{y:.2f} Z<extra></extra>',
        ))

        # HbR shaded std band using a single filled area trace
        hbr_upper = (hbr_mean + hbr_std).tolist()
        hbr_lower = (hbr_mean - hbr_std).tolist()
        
        fig.add_trace(go.Scatter(
            x=times_list + times_list[::-1],
            y=hbr_upper + hbr_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.15)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='HbR std band',
        ))

        # HbR mean trace
        fig.add_trace(go.Scatter(
            x=times,
            y=hbr_mean,
            mode='lines',
            name=f'HbR (n={len(hbr_indices)} ch)',
            line=dict(color='blue', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>HbR: %{y:.2f} Z<extra></extra>',
        ))

        # Add stimulus markers as vertical lines
        events, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        event_colors = {'LEFT': '#2ecc71', 'RIGHT': '#9b59b6', 'NOTHING': '#95a5a6'}

        # Group events by type for legend
        events_added = {'LEFT': False, 'RIGHT': False, 'NOTHING': False}

        for event in events:
            onset_sample = event[0]
            if onset_sample >= len(times):
                continue
            onset_time = times[onset_sample]
            event_code = event[2]

            # Find label
            label = None
            for k, v in event_id.items():
                if v == event_code:
                    label = k
                    break

            if label:
                base_label = label.split('/')[0] if '/' in label else label
                color = event_colors.get(base_label, 'gray')

                # Add vertical line for stimulus (thick for visibility)
                fig.add_vline(
                    x=onset_time,
                    line=dict(color=color, width=3, dash='dash'),
                    opacity=0.7,
                )

                # Track which event types have been added
                if not events_added.get(base_label, True):
                    events_added[base_label] = True

        # Add invisible traces for stimulus legend
        for label, color in event_colors.items():
            if events_added.get(label, False):
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    name=f'{label} stimulus',
                    line=dict(color=color, width=2, dash='dash'),
                ))

        # Calculate initial view window (first 60 seconds or full duration if shorter)
        total_duration = times[-1] - times[0]
        initial_window = min(60.0, total_duration)
        initial_range = [times[0], times[0] + initial_window]

        # Configure layout with range slider
        fig.update_layout(
            title=dict(
                text=f'fNIRS Time Series (Z-scores) - Subject {config.subject.id}<br>'
                     f'<sup>Mean ± SD across good channels | Use slider or arrow keys to navigate</sup>',
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Time (s)',
                range=initial_range,
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                    bgcolor='#f0f0f0',
                    bordercolor='#ccc',
                    borderwidth=1,
                ),
                type='linear',
            ),
            yaxis=dict(
                title='Z-score',
                range=[-3, 3],
                fixedrange=False,
            ),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1.0,
                xanchor='left',
                x=1.02,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1,
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(t=100, b=80, r=180),
        )

        # Custom JavaScript for arrow key navigation
        arrow_key_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                        var xaxis = plotDiv._fullLayout.xaxis;
                        var currentRange = xaxis.range;
                        var rangeWidth = currentRange[1] - currentRange[0];
                        var step = rangeWidth * 0.1;  // 10% of visible range
                        
                        var newRange;
                        if (e.key === 'ArrowLeft') {
                            newRange = [currentRange[0] - step, currentRange[1] - step];
                        } else {
                            newRange = [currentRange[0] + step, currentRange[1] + step];
                        }
                        
                        // Clamp to data bounds
                        var dataRange = xaxis._rangeInitial || [0, currentRange[1]];
                        if (newRange[0] < 0) {
                            newRange = [0, rangeWidth];
                        }
                        
                        Plotly.relayout(plotDiv, {'xaxis.range': newRange});
                        e.preventDefault();
                    }
                });
            }
        });
        </script>
        """

        # Save as interactive HTML with custom JS
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_timeseries.html"
        )
        filepath = output_path / filename
        
        # Write HTML with embedded JS for arrow key navigation
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        # Insert custom JS before closing body tag
        html_content = html_content.replace('</body>', f'{arrow_key_js}</body>')
        
        with open(filepath, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        logger.info(f"Interactive fNIRS time series plot saved to: {filepath}")
        return filepath

    except Exception as exc:
        logger.error(f"Failed to generate fNIRS time series plot: {exc}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_timeseries_plot_by_roi(
    raw_haemo: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
    roi_name: str,
    source_prefixes: list[str],
    good_channels: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Generate interactive HbO/HbR time series plot for a specific ROI in Z-scores.

    Creates an interactive HTML plot using Plotly for channels belonging
    to a specific ROI (defined by source prefixes). Shows individual channel
    traces (thin/light) plus the mean trace (thick/solid).

    Features:
    - Z-scored data with ylim [-3, 3]
    - Individual channel traces (thin, lighter color)
    - Mean trace (thick, solid color)
    - Shaded std band around mean

    Args:
        raw_haemo: Preprocessed hemoglobin data (hbo/hbr channels).
        output_path: Directory to save the plot.
        config: Subject configuration.
        roi_name: Name of the ROI (e.g., "Left Anterior").
        source_prefixes: List of source prefixes for this ROI (e.g., ["S1_", "S9_"]).
        good_channels: List of good channel pairs (e.g., ["S1_D1"]).
                       If None, uses all channels matching the ROI.

    Returns:
        Path to saved HTML file, or None if no channels for this ROI.
    """
    import plotly.graph_objects as go
    from scipy.stats import zscore

    # Convert ROI name to filename-safe format
    roi_slug = roi_name.lower().replace(" ", "_")
    logger.info(f"Generating interactive fNIRS time series plot for ROI: {roi_name}...")

    try:
        data = raw_haemo.get_data()
        times = raw_haemo.times
        ch_names = raw_haemo.ch_names

        # Filter channels for this ROI
        hbo_indices = []
        hbr_indices = []
        hbo_names = []
        hbr_names = []

        for idx, ch_name in enumerate(ch_names):
            # Check if channel matches ROI source prefixes
            matches_roi = any(prefix in ch_name for prefix in source_prefixes)
            if not matches_roi:
                continue

            # Check if channel is in good_channels list
            base_pair = ch_name.split(" ")[0]
            if good_channels is not None and base_pair not in good_channels:
                continue

            if "hbo" in ch_name:
                hbo_indices.append(idx)
                hbo_names.append(ch_name.replace(" hbo", ""))
            elif "hbr" in ch_name:
                hbr_indices.append(idx)
                hbr_names.append(ch_name.replace(" hbr", ""))

        # Handle edge case: no channels for this ROI
        if not hbo_indices and not hbr_indices:
            logger.warning(f"No good channels available for ROI '{roi_name}' time series plot")
            return None

        logger.info(
            f"ROI '{roi_name}': {len(hbo_indices)} HbO and {len(hbr_indices)} HbR channels"
        )

        fig = go.Figure()
        times_list = times.tolist()

        # Z-score each channel individually
        if hbo_indices:
            hbo_zscored = np.array([zscore(data[idx, :]) for idx in hbo_indices])
            hbo_mean = np.mean(hbo_zscored, axis=0)
            hbo_std = np.std(hbo_zscored, axis=0)
            hbo_upper = (hbo_mean + hbo_std).tolist()
            hbo_lower = (hbo_mean - hbo_std).tolist()

            # Add individual HbO channel traces (thin, lighter color)
            for ch_idx, ch_name in enumerate(hbo_names):
                fig.add_trace(go.Scatter(
                    x=times,
                    y=hbo_zscored[ch_idx, :],
                    mode='lines',
                    name=f'HbO {ch_name}',
                    line=dict(color='rgba(255, 100, 100, 0.4)', width=0.8),
                    hovertemplate=f'{ch_name}<br>Time: %{{x:.2f}}s<br>HbO: %{{y:.2f}} Z<extra></extra>',
                    legendgroup='hbo_individual',
                    showlegend=(ch_idx == 0),
                    legendgrouptitle_text='HbO channels' if ch_idx == 0 else None,
                ))

            # HbO std band
            fig.add_trace(go.Scatter(
                x=times_list + times_list[::-1],
                y=hbo_upper + hbo_lower[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.15)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                name='HbO std band',
            ))

            # HbO mean trace (thick, solid)
            fig.add_trace(go.Scatter(
                x=times,
                y=hbo_mean,
                mode='lines',
                name=f'HbO Mean (n={len(hbo_indices)})',
                line=dict(color='red', width=2.5),
                hovertemplate='Time: %{x:.2f}s<br>HbO Mean: %{y:.2f} Z<extra></extra>',
            ))

        if hbr_indices:
            hbr_zscored = np.array([zscore(data[idx, :]) for idx in hbr_indices])
            hbr_mean = np.mean(hbr_zscored, axis=0)
            hbr_std = np.std(hbr_zscored, axis=0)
            hbr_upper = (hbr_mean + hbr_std).tolist()
            hbr_lower = (hbr_mean - hbr_std).tolist()

            # Add individual HbR channel traces (thin, lighter color)
            for ch_idx, ch_name in enumerate(hbr_names):
                fig.add_trace(go.Scatter(
                    x=times,
                    y=hbr_zscored[ch_idx, :],
                    mode='lines',
                    name=f'HbR {ch_name}',
                    line=dict(color='rgba(100, 100, 255, 0.4)', width=0.8),
                    hovertemplate=f'{ch_name}<br>Time: %{{x:.2f}}s<br>HbR: %{{y:.2f}} Z<extra></extra>',
                    legendgroup='hbr_individual',
                    showlegend=(ch_idx == 0),
                    legendgrouptitle_text='HbR channels' if ch_idx == 0 else None,
                ))

            # HbR std band
            fig.add_trace(go.Scatter(
                x=times_list + times_list[::-1],
                y=hbr_upper + hbr_lower[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.15)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                name='HbR std band',
            ))

            # HbR mean trace (thick, solid)
            fig.add_trace(go.Scatter(
                x=times,
                y=hbr_mean,
                mode='lines',
                name=f'HbR Mean (n={len(hbr_indices)})',
                line=dict(color='blue', width=2.5),
                hovertemplate='Time: %{x:.2f}s<br>HbR Mean: %{y:.2f} Z<extra></extra>',
            ))

        # Add stimulus markers
        events, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        event_colors = {'LEFT': '#2ecc71', 'RIGHT': '#9b59b6', 'NOTHING': '#95a5a6'}
        events_added = {'LEFT': False, 'RIGHT': False, 'NOTHING': False}

        for event in events:
            onset_sample = event[0]
            if onset_sample >= len(times):
                continue
            onset_time = times[onset_sample]
            event_code = event[2]

            label = None
            for k, v in event_id.items():
                if v == event_code:
                    label = k
                    break

            if label:
                base_label = label.split('/')[0] if '/' in label else label
                color = event_colors.get(base_label, 'gray')

                fig.add_vline(
                    x=onset_time,
                    line=dict(color=color, width=3, dash='dash'),
                    opacity=0.7,
                )

                if not events_added.get(base_label, True):
                    events_added[base_label] = True

        # Add stimulus legend traces
        for label, color in event_colors.items():
            if events_added.get(label, False):
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    name=f'{label} stimulus',
                    line=dict(color=color, width=2, dash='dash'),
                    legendgroup='stimuli',
                ))

        # Calculate initial view window
        total_duration = times[-1] - times[0]
        initial_window = min(60.0, total_duration)
        initial_range = [times[0], times[0] + initial_window]

        # Configure layout
        fig.update_layout(
            title=dict(
                text=f'fNIRS Time Series (Z-scores) - {roi_name} - Subject {config.subject.id}<br>'
                     f'<sup>Individual channels (thin) + Mean ± SD (thick) | Use slider to navigate</sup>',
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Time (s)',
                range=initial_range,
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                    bgcolor='#f0f0f0',
                    bordercolor='#ccc',
                    borderwidth=1,
                ),
                type='linear',
            ),
            yaxis=dict(
                title='Z-score',
                range=[-3, 3],
                fixedrange=False,
            ),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1.0,
                xanchor='left',
                x=1.02,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1,
                groupclick='toggleitem',
                tracegroupgap=5,
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(t=100, b=80, r=200),
        )

        # Arrow key navigation JS
        arrow_key_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                        var xaxis = plotDiv._fullLayout.xaxis;
                        var currentRange = xaxis.range;
                        var rangeWidth = currentRange[1] - currentRange[0];
                        var step = rangeWidth * 0.1;
                        
                        var newRange;
                        if (e.key === 'ArrowLeft') {
                            newRange = [currentRange[0] - step, currentRange[1] - step];
                        } else {
                            newRange = [currentRange[0] + step, currentRange[1] + step];
                        }
                        
                        if (newRange[0] < 0) {
                            newRange = [0, rangeWidth];
                        }
                        
                        Plotly.relayout(plotDiv, {'xaxis.range': newRange});
                        e.preventDefault();
                    }
                });
            }
        });
        </script>
        """

        # Save HTML
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_timeseries_{roi_slug}.html"
        )
        filepath = output_path / filename

        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html_content = html_content.replace('</body>', f'{arrow_key_js}</body>')

        with open(filepath, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        logger.info(f"ROI time series plot saved to: {filepath}")
        return filepath

    except Exception as exc:
        logger.error(f"Failed to generate ROI time series plot for '{roi_name}': {exc}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_timeseries_all_rois(
    raw_haemo: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
    good_channels: Optional[list[str]] = None,
) -> Optional[Path]:
    """Generate interactive time series plot combining all 4 ROIs in Z-scores.

    Creates an interactive HTML plot using Plotly showing HbO mean traces
    for all 4 ROIs (Left/Right Anterior, Left/Right Posterior) overlaid
    on the same plot with different colors for easy comparison.

    Features:
    - Z-scored data with ylim [-3, 3]
    - 4 ROI traces with distinct colors
    - Shaded std bands for each ROI
    - Stimulus markers (LEFT/RIGHT/NOTHING)
    - Range slider for navigation
    - Arrow key navigation (left/right)

    Args:
        raw_haemo: Preprocessed hemoglobin data (hbo/hbr channels).
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_channels: Optional list of good channel pairs (e.g., ["S1_D1"]).

    Returns:
        Path to saved HTML file, or None if failed.
    """
    import plotly.graph_objects as go
    from scipy.stats import zscore

    logger.info("Generating interactive fNIRS time series plot (all 4 ROIs combined, Z-scores)...")

    try:
        data = raw_haemo.get_data()
        times = raw_haemo.times
        ch_names = raw_haemo.ch_names
        times_list = times.tolist()

        # ROI definitions with colors
        roi_config = {
            'Left Anterior': {
                'prefixes': ['S1_', 'S9_'],
                'color': 'rgba(231, 76, 60, 1)',      # Red
                'fill_color': 'rgba(231, 76, 60, 0.15)',
            },
            'Right Anterior': {
                'prefixes': ['S2_', 'S10_'],
                'color': 'rgba(46, 204, 113, 1)',     # Green
                'fill_color': 'rgba(46, 204, 113, 0.15)',
            },
            'Left Posterior': {
                'prefixes': ['S3_', 'S4_', 'S5_'],
                'color': 'rgba(52, 152, 219, 1)',     # Blue
                'fill_color': 'rgba(52, 152, 219, 0.15)',
            },
            'Right Posterior': {
                'prefixes': ['S6_', 'S7_', 'S8_'],
                'color': 'rgba(155, 89, 182, 1)',     # Purple
                'fill_color': 'rgba(155, 89, 182, 0.15)',
            },
        }

        fig = go.Figure()

        # Add traces for each ROI
        for roi_name, roi_info in roi_config.items():
            # Find HbO channels for this ROI
            hbo_indices = []
            for ch_idx, ch_name in enumerate(ch_names):
                if 'hbo' not in ch_name.lower():
                    continue

                # Check if channel matches ROI prefixes
                matches_roi = any(prefix in ch_name for prefix in roi_info['prefixes'])
                if not matches_roi:
                    continue

                # Check if channel is in good_channels
                if good_channels is not None:
                    base_name = ch_name.split(' ')[0]
                    if base_name not in good_channels:
                        continue

                hbo_indices.append(ch_idx)

            if not hbo_indices:
                logger.warning(f"ROI '{roi_name}': No good HbO channels found, skipping")
                continue

            # Z-score each channel individually, then compute mean
            hbo_zscored = np.array([zscore(data[idx, :]) for idx in hbo_indices])
            hbo_mean = np.mean(hbo_zscored, axis=0)
            hbo_std = np.std(hbo_zscored, axis=0)
            hbo_upper = (hbo_mean + hbo_std).tolist()
            hbo_lower = (hbo_mean - hbo_std).tolist()

            # Std band (shaded area)
            fig.add_trace(go.Scatter(
                x=times_list + times_list[::-1],
                y=hbo_upper + hbo_lower[::-1],
                fill='toself',
                fillcolor=roi_info['fill_color'],
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                name=f'{roi_name} std',
            ))

            # Mean trace
            fig.add_trace(go.Scatter(
                x=times,
                y=hbo_mean,
                mode='lines',
                name=f'{roi_name} (n={len(hbo_indices)})',
                line=dict(color=roi_info['color'], width=2),
                hovertemplate=f'{roi_name}<br>Time: %{{x:.2f}}s<br>HbO: %{{y:.2f}} Z<extra></extra>',
            ))

        # Add stimulus markers
        events, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        event_colors = {'LEFT': '#27ae60', 'RIGHT': '#8e44ad', 'NOTHING': '#7f8c8d'}
        events_added = {'LEFT': False, 'RIGHT': False, 'NOTHING': False}

        for event in events:
            onset_sample = event[0]
            if onset_sample >= len(times):
                continue
            onset_time = times[onset_sample]
            event_code = event[2]

            for event_name, code in event_id.items():
                if code == event_code:
                    base_name = event_name.split('/')[0] if '/' in event_name else event_name
                    if base_name in event_colors:
                        show_legend = not events_added.get(base_name, False)
                        fig.add_vline(
                            x=onset_time,
                            line=dict(color=event_colors[base_name], width=3, dash='dash'),
                            opacity=0.7,
                        )
                        if show_legend:
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='lines',
                                name=base_name,
                                line=dict(color=event_colors[base_name], width=3, dash='dash'),
                                showlegend=True,
                            ))
                            events_added[base_name] = True
                    break

        # Initial view window (60 seconds)
        initial_window = min(60.0, times[-1])

        fig.update_layout(
            title=dict(
                text=f"fNIRS HbO Time Series (Z-scores) - All ROIs Combined<br>"
                     f"<sub>Subject {config.subject.id} | Session {config.subject.session}</sub>",
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Time (s)',
                range=[0, initial_window],
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                    bgcolor='#f0f0f0',
                    bordercolor='#ccc',
                    borderwidth=1,
                ),
                type='linear',
            ),
            yaxis=dict(
                title='Z-score',
                range=[-3, 3],
                fixedrange=False,
            ),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1.0,
                xanchor='left',
                x=1.02,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1,
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(t=100, b=80, r=180),
        )

        # Arrow key navigation JS
        arrow_key_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                        var xaxis = plotDiv._fullLayout.xaxis;
                        var currentRange = xaxis.range;
                        var rangeWidth = currentRange[1] - currentRange[0];
                        var step = rangeWidth * 0.1;
                        
                        var newRange;
                        if (e.key === 'ArrowLeft') {
                            newRange = [currentRange[0] - step, currentRange[1] - step];
                        } else {
                            newRange = [currentRange[0] + step, currentRange[1] + step];
                        }
                        
                        if (newRange[0] < 0) {
                            newRange = [0, rangeWidth];
                        }
                        
                        Plotly.relayout(plotDiv, {'xaxis.range': newRange});
                        e.preventDefault();
                    }
                });
            }
        });
        </script>
        """

        # Save HTML
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_timeseries_all_rois.html"
        )
        filepath = output_path / filename

        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html_content = html_content.replace('</body>', f'{arrow_key_js}</body>')

        with open(filepath, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        logger.info(f"All ROIs time series plot saved to: {filepath}")
        return filepath

    except Exception as exc:
        logger.error(f"Failed to generate all ROIs time series plot: {exc}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_contrast_map_anterior(
    fnirs_epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate contrast maps using only ANTERIOR ROIs for fNIRS lateralization analysis.
    
    Creates a 2x3 figure with HbO (top row) and HbR (bottom row) analysis:
    - Column 1: Amplitude by condition and hemisphere
    - Column 2: Lateralization index
    - Column 3: Motor vs Rest contrast
    
    Uses only anterior channels (motor cortex region) for cleaner motor-related signal:
    - Left Anterior: S1_, S9_ (left motor cortex)
    - Right Anterior: S2_, S10_ (right motor cortex)
    
    Scientific rationale:
        - LEFT hand movement → Right hemisphere activation (contralateral)
        - RIGHT hand movement → Left hemisphere activation (contralateral)
        - HbO increases and HbR decreases indicate neural activation
        - Anterior ROIs capture primary motor cortex activity more directly
    
    Args:
        fnirs_epochs: MNE Epochs with fNIRS data
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved figure or None if failed
    """
    try:
        logger.info("Generating fNIRS contrast map (anterior ROIs only, HbO + HbR)...")
        
        conditions = list(fnirs_epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        # Get HbO and HbR channels
        hbo_channels = [ch for ch in fnirs_epochs.ch_names if 'hbo' in ch.lower()]
        hbr_channels = [ch for ch in fnirs_epochs.ch_names if 'hbr' in ch.lower()]
        
        if not hbo_channels or not hbr_channels:
            logger.warning("No HbO or HbR channels found")
            return None
        
        # Define ANTERIOR hemispheres only (motor cortex region)
        # Left Anterior: S1_, S9_
        # Right Anterior: S2_, S10_
        left_anterior_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S1_', 'S9_'])]
        right_anterior_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S2_', 'S10_'])]
        left_anterior_hbr = [ch for ch in hbr_channels if any(x in ch for x in ['S1_', 'S9_'])]
        right_anterior_hbr = [ch for ch in hbr_channels if any(x in ch for x in ['S2_', 'S10_'])]
        
        logger.info(f"Left Anterior HbO channels: {left_anterior_hbo}")
        logger.info(f"Right Anterior HbO channels: {right_anterior_hbo}")
        logger.info(f"Left Anterior HbR channels: {left_anterior_hbr}")
        logger.info(f"Right Anterior HbR channels: {right_anterior_hbr}")
        
        if not left_anterior_hbo or not right_anterior_hbo:
            logger.warning("Could not find anterior ROI channels, falling back to all channels")
            mid = len(hbo_channels) // 2
            left_anterior_hbo = hbo_channels[:mid] if mid > 0 else hbo_channels[:1]
            right_anterior_hbo = hbo_channels[mid:] if mid > 0 else hbo_channels[1:2]
            mid = len(hbr_channels) // 2
            left_anterior_hbr = hbr_channels[:mid] if mid > 0 else hbr_channels[:1]
            right_anterior_hbr = hbr_channels[mid:] if mid > 0 else hbr_channels[1:2]
        
        times = fnirs_epochs.times
        # Task window: from onset to end of task
        task_mask = (times >= 2) & (times <= config.trials.task_duration_sec + 5)
        
        # Get condition names
        left_cond = [c for c in conditions if 'LEFT' in c][0] if has_left else None
        right_cond = [c for c in conditions if 'RIGHT' in c][0] if has_right else None
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Helper function to compute results for a chromophore
        def compute_chromophore_results(
            left_channels: list[str],
            right_channels: list[str],
        ) -> dict[str, float]:
            """Compute mean amplitude and std for each condition and hemisphere."""
            results = {}
            for cond_name, cond_key in [('LEFT', left_cond), ('RIGHT', right_cond), ('NOTHING', nothing_cond)]:
                if cond_key is None:
                    continue
                
                epochs_cond = fnirs_epochs[cond_key]
                
                # Left Anterior hemisphere
                if left_channels:
                    data = epochs_cond.get_data(picks=left_channels)[:, :, task_mask]
                    results[f'{cond_name}_left'] = data.mean() * 1e6
                    results[f'{cond_name}_left_std'] = data.mean(axis=(1, 2)).std() * 1e6
                
                # Right Anterior hemisphere
                if right_channels:
                    data = epochs_cond.get_data(picks=right_channels)[:, :, task_mask]
                    results[f'{cond_name}_right'] = data.mean() * 1e6
                    results[f'{cond_name}_right_std'] = data.mean(axis=(1, 2)).std() * 1e6
            
            return results
        
        # Compute results for HbO and HbR
        results_hbo = compute_chromophore_results(left_anterior_hbo, right_anterior_hbo)
        results_hbr = compute_chromophore_results(left_anterior_hbr, right_anterior_hbr)
        
        # Create figure: 2 rows (HbO, HbR) x 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Helper function to plot a row for a chromophore
        def plot_chromophore_row(
            row_axes: np.ndarray,
            results: dict[str, float],
            chromophore_name: str,
            chromophore_color: str,
        ) -> None:
            """Plot the 3 panels for a single chromophore."""
            
            # Plot 1: Condition comparison by hemisphere
            ax = row_axes[0]
            x = np.arange(2)
            width = 0.25
            
            conditions_to_plot = []
            if 'LEFT_left' in results:
                conditions_to_plot.append(('LEFT', 'blue'))
            if 'RIGHT_left' in results:
                conditions_to_plot.append(('RIGHT', 'red'))
            if 'NOTHING_left' in results:
                conditions_to_plot.append(('NOTHING', 'green'))
            
            for i, (cond, color) in enumerate(conditions_to_plot):
                left_val = results.get(f'{cond}_left', 0)
                right_val = results.get(f'{cond}_right', 0)
                left_err = results.get(f'{cond}_left_std', 0)
                right_err = results.get(f'{cond}_right_std', 0)
                
                ax.bar(x + i * width, [left_val, right_val], width, 
                      label=cond, color=color, alpha=0.7,
                      yerr=[left_err, right_err], capsize=3)
            
            ax.set_xticks(x + width)
            ax.set_xticklabels(['Left Anterior', 'Right Anterior'])
            ax.set_ylabel(f'Mean Δ{chromophore_name} (μM)', fontsize=11)
            ax.set_title(f'{chromophore_name} Amplitude by Condition\n(Anterior ROIs Only)', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Lateralization Index
            ax = row_axes[1]
            lat_indices = []
            lat_labels = []
            lat_colors = []
            
            for cond, color in [('LEFT', 'blue'), ('RIGHT', 'red'), ('NOTHING', 'green')]:
                if f'{cond}_left' in results and f'{cond}_right' in results:
                    left_val = results[f'{cond}_left']
                    right_val = results[f'{cond}_right']
                    if abs(left_val) + abs(right_val) > 0:
                        lat_idx = (right_val - left_val) / (abs(right_val) + abs(left_val))
                    else:
                        lat_idx = 0
                    lat_indices.append(lat_idx)
                    lat_labels.append(cond)
                    lat_colors.append(color)
            
            if lat_indices:
                ax.bar(lat_labels, lat_indices, color=lat_colors, alpha=0.7)
                ax.axhline(0, color='black', linestyle='-', linewidth=1)
                ax.set_ylabel(f'{chromophore_name} Lateralization Index\n(R-L)/(|R|+|L|)', fontsize=11)
                ax.set_title(f'{chromophore_name} Hemispheric Lateralization\n(+) = Right, (-) = Left', 
                            fontsize=12, fontweight='bold')
                ax.set_ylim(-1, 1)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add expected pattern annotation
                expected_text = 'Expected:\nLEFT hand → (+) Right\nRIGHT hand → (-) Left'
                ax.text(0.02, 0.98, expected_text,
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 3: Motor vs Rest contrast
            ax = row_axes[2]
            if has_nothing and (has_left or has_right):
                motor_conditions = []
                if 'LEFT_left' in results:
                    motor_conditions.append(('LEFT', results['LEFT_left'], results['LEFT_right']))
                if 'RIGHT_left' in results:
                    motor_conditions.append(('RIGHT', results['RIGHT_left'], results['RIGHT_right']))
                
                if motor_conditions:
                    motor_left = np.mean([c[1] for c in motor_conditions])
                    motor_right = np.mean([c[2] for c in motor_conditions])
                    nothing_left = results.get('NOTHING_left', 0)
                    nothing_right = results.get('NOTHING_right', 0)
                    
                    x = np.arange(2)
                    width = 0.35
                    
                    ax.bar(x - width/2, [motor_left, motor_right], width, 
                          label='Motor (L+R avg)', color='purple', alpha=0.7)
                    ax.bar(x + width/2, [nothing_left, nothing_right], width,
                          label='NOTHING', color='gray', alpha=0.7)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(['Left Anterior', 'Right Anterior'])
                    ax.set_ylabel(f'Mean Δ{chromophore_name} (μM)', fontsize=11)
                    ax.set_title(f'{chromophore_name} Motor vs Rest\n(Sanity Check)', 
                                fontsize=12, fontweight='bold')
                    ax.legend(loc='best', fontsize=8)
                    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
                    ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'NOTHING condition\nnot available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{chromophore_name} Motor vs Rest', fontsize=12, fontweight='bold')
        
        # Plot HbO row (top)
        plot_chromophore_row(axes[0, :], results_hbo, 'HbO', 'red')
        
        # Plot HbR row (bottom)
        plot_chromophore_row(axes[1, :], results_hbr, 'HbR', 'blue')
        
        fig.suptitle(f'fNIRS Contrast Analysis (Anterior ROIs)\nSubject {config.subject.id}',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_contrast_anterior.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"fNIRS contrast map (anterior, HbO+HbR) saved to: {filepath}")
        return filepath
        
    except Exception as exc:
        logger.error(f"Failed to generate fNIRS contrast map (anterior): {exc}")
        import traceback
        traceback.print_exc()
        return None


def generate_multimodal_timeseries_plot(
    raw_haemo: mne.io.Raw,
    raw_eeg: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
    good_fnirs_channels: Optional[list[str]] = None,
    eeg_channels: Optional[list[str]] = None,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
) -> Optional[Path]:
    """
    Generate interactive multimodal time series plot with fNIRS and EEG band power.

    Creates an interactive HTML plot using Plotly with:
    - HbO (red) and HbR (blue) traces from fNIRS
    - Alpha power (orange) and Beta power (green) from EEG
    - All signals z-scored for comparable visualization
    - Horizontal range slider for navigation

    The EEG band power is computed using the Hilbert transform method:
    1. Bandpass filter EEG to the frequency band of interest
    2. Apply Hilbert transform to get analytic signal
    3. Compute instantaneous power as |analytic_signal|²
    4. Smooth with moving average for visualization

    Args:
        raw_haemo: Preprocessed hemoglobin data (hbo/hbr channels).
        raw_eeg: Preprocessed EEG data.
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_fnirs_channels: List of good fNIRS channel pairs.
        eeg_channels: List of EEG channels to use. If None, uses C3, C4, Cz.
        alpha_band: Alpha frequency band (default 8-13 Hz).
        beta_band: Beta frequency band (default 13-30 Hz).

    Returns:
        Path to saved HTML file, or None on error.

    Scientific Reference:
        Hilbert transform for instantaneous amplitude/power estimation is
        standard in neurophysiology. See: Bruns (2004) J Neurosci Methods.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import hilbert
    from scipy.ndimage import uniform_filter1d

    logger.info("Generating multimodal fNIRS + EEG time series plot...")

    try:
        # =====================================================================
        # fNIRS Processing
        # =====================================================================
        fnirs_data = raw_haemo.get_data()
        fnirs_times = raw_haemo.times
        fnirs_ch_names = raw_haemo.ch_names

        # Filter fNIRS channels
        if good_fnirs_channels is not None:
            hbo_indices = [
                idx for idx, name in enumerate(fnirs_ch_names)
                if name.split(" ")[0] in good_fnirs_channels and "hbo" in name
            ]
            hbr_indices = [
                idx for idx, name in enumerate(fnirs_ch_names)
                if name.split(" ")[0] in good_fnirs_channels and "hbr" in name
            ]
        else:
            hbo_indices = [idx for idx, name in enumerate(fnirs_ch_names) if "hbo" in name]
            hbr_indices = [idx for idx, name in enumerate(fnirs_ch_names) if "hbr" in name]

        if not hbo_indices or not hbr_indices:
            logger.warning("No good fNIRS channels for multimodal plot")
            return None

        # Compute mean HbO/HbR (convert to μM)
        hbo_mean = np.mean(fnirs_data[hbo_indices, :], axis=0) * 1e6
        hbr_mean = np.mean(fnirs_data[hbr_indices, :], axis=0) * 1e6

        # =====================================================================
        # EEG Processing - Band Power via Hilbert Transform
        # =====================================================================
        # Select EEG channels (default: sensorimotor channels)
        if eeg_channels is None:
            eeg_channels = ['C3', 'C4', 'Cz', 'CP3', 'CP4']

        # Find available channels
        available_eeg = [ch for ch in eeg_channels if ch in raw_eeg.ch_names]
        if not available_eeg:
            # Fallback: use all EEG channels
            available_eeg = [ch for ch in raw_eeg.ch_names if ch not in raw_eeg.info['bads']][:10]
            logger.warning(f"Requested EEG channels not found, using: {available_eeg[:5]}...")

        if not available_eeg:
            logger.warning("No EEG channels available for multimodal plot")
            return None

        logger.info(f"Using EEG channels: {available_eeg}")

        # Get EEG data and sampling rate
        eeg_sfreq = raw_eeg.info['sfreq']
        raw_eeg_subset = raw_eeg.copy().pick(available_eeg)
        eeg_data = raw_eeg_subset.get_data()
        eeg_times = raw_eeg_subset.times

        def compute_band_power_hilbert(
            data: np.ndarray,
            sfreq: float,
            freq_band: tuple[float, float],
            smoothing_samples: int = 500,
        ) -> np.ndarray:
            """Compute instantaneous band power using Hilbert transform.

            Args:
                data: EEG data array (n_channels, n_times).
                sfreq: Sampling frequency in Hz.
                freq_band: Tuple of (low_freq, high_freq) in Hz.
                smoothing_samples: Window size for moving average smoothing.

            Returns:
                Mean band power across channels (n_times,).
            """
            from mne.filter import filter_data

            # Bandpass filter
            filtered = filter_data(
                data,
                sfreq,
                l_freq=freq_band[0],
                h_freq=freq_band[1],
                verbose=False,
            )

            # Hilbert transform for analytic signal
            analytic = hilbert(filtered, axis=1)

            # Instantaneous power = |analytic|²
            power = np.abs(analytic) ** 2

            # Mean across channels
            mean_power = np.mean(power, axis=0)

            # Smooth for visualization
            smoothed = uniform_filter1d(mean_power, size=smoothing_samples)

            return smoothed

        # Compute alpha and beta power
        smoothing_window = int(eeg_sfreq * 0.5)  # 500ms smoothing
        alpha_power = compute_band_power_hilbert(eeg_data, eeg_sfreq, alpha_band, smoothing_window)
        beta_power = compute_band_power_hilbert(eeg_data, eeg_sfreq, beta_band, smoothing_window)

        # =====================================================================
        # Resample EEG to fNIRS time base (fNIRS is typically lower sfreq)
        # =====================================================================
        from scipy.interpolate import interp1d

        # Interpolate EEG power to fNIRS time points
        alpha_interp = interp1d(eeg_times, alpha_power, kind='linear', fill_value='extrapolate')
        beta_interp = interp1d(eeg_times, beta_power, kind='linear', fill_value='extrapolate')

        alpha_resampled = alpha_interp(fnirs_times)
        beta_resampled = beta_interp(fnirs_times)

        # =====================================================================
        # Robust Z-score normalization for comparable visualization
        # Uses median and MAD to handle outliers
        # =====================================================================
        def robust_zscore(arr: np.ndarray, clip_threshold: float = 5.0) -> np.ndarray:
            """Robust z-score normalize using median and MAD.
            
            Standard z-score is sensitive to outliers. This uses:
            - Median instead of mean (robust to outliers)
            - MAD (Median Absolute Deviation) instead of std
            - Clipping to ±clip_threshold to prevent extreme values
            
            Args:
                arr: Input array to normalize.
                clip_threshold: Values beyond ±threshold are clipped.
                
            Returns:
                Robust z-scored array with outliers handled.
            """
            median = np.median(arr)
            # MAD = median(|x - median(x)|)
            mad = np.median(np.abs(arr - median))
            # Scale MAD to be comparable to std for normal distribution
            # For normal distribution: std ≈ 1.4826 * MAD
            mad_scaled = mad * 1.4826
            
            if mad_scaled < 1e-10:
                # Fallback to standard zscore if MAD is zero
                std = np.std(arr)
                if std < 1e-10:
                    return np.zeros_like(arr)
                z = (arr - np.mean(arr)) / std
            else:
                z = (arr - median) / mad_scaled
            
            # Clip extreme values
            z = np.clip(z, -clip_threshold, clip_threshold)
            
            return z

        hbo_z = robust_zscore(hbo_mean)
        hbr_z = robust_zscore(hbr_mean)
        alpha_z = robust_zscore(alpha_resampled)
        beta_z = robust_zscore(beta_resampled)

        # =====================================================================
        # Create Plotly Figure with Dual Y-Axes
        # =====================================================================
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'fNIRS: HbO & HbR (z-scored)',
                'EEG: Alpha & Beta Power (z-scored)'
            ),
            row_heights=[0.5, 0.5],
        )

        times_list = fnirs_times.tolist()

        # fNIRS traces (row 1)
        fig.add_trace(go.Scatter(
            x=fnirs_times,
            y=hbo_z,
            mode='lines',
            name=f'HbO (n={len(hbo_indices)} ch)',
            line=dict(color='red', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>HbO (z): %{y:.2f}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=fnirs_times,
            y=hbr_z,
            mode='lines',
            name=f'HbR (n={len(hbr_indices)} ch)',
            line=dict(color='blue', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>HbR (z): %{y:.2f}<extra></extra>',
        ), row=1, col=1)

        # EEG traces (row 2)
        fig.add_trace(go.Scatter(
            x=fnirs_times,
            y=alpha_z,
            mode='lines',
            name=f'Alpha ({alpha_band[0]}-{alpha_band[1]} Hz)',
            line=dict(color='orange', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>Alpha (z): %{y:.2f}<extra></extra>',
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=fnirs_times,
            y=beta_z,
            mode='lines',
            name=f'Beta ({beta_band[0]}-{beta_band[1]} Hz)',
            line=dict(color='green', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>Beta (z): %{y:.2f}<extra></extra>',
        ), row=2, col=1)

        # Add stimulus markers to both subplots
        events, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        event_colors = {'LEFT': '#2ecc71', 'RIGHT': '#9b59b6', 'NOTHING': '#95a5a6'}

        for event in events:
            onset_sample = event[0]
            if onset_sample >= len(fnirs_times):
                continue
            onset_time = fnirs_times[onset_sample]
            event_code = event[2]

            label = None
            for k, v in event_id.items():
                if v == event_code:
                    label = k
                    break

            if label:
                base_label = label.split('/')[0] if '/' in label else label
                color = event_colors.get(base_label, 'gray')

                # Add to both rows (thick lines for visibility)
                for row in [1, 2]:
                    fig.add_vline(
                        x=onset_time,
                        line=dict(color=color, width=3, dash='dash'),
                        opacity=0.7,
                        row=row, col=1,
                    )

        # Calculate initial view window
        total_duration = fnirs_times[-1] - fnirs_times[0]
        initial_window = min(60.0, total_duration)
        initial_range = [fnirs_times[0], fnirs_times[0] + initial_window]

        # Configure layout
        fig.update_layout(
            title=dict(
                text=f'Multimodal Time Series - Subject {config.subject.id}<br>'
                     f'<sup>fNIRS + EEG Band Power | Use slider or arrow keys to navigate</sup>',
                x=0.5,
                xanchor='center',
            ),
            xaxis2=dict(
                title='Time (s)',
                range=initial_range,
                rangeslider=dict(
                    visible=True,
                    thickness=0.06,
                    bgcolor='#f0f0f0',
                ),
            ),
            yaxis=dict(title='Z-score'),
            yaxis2=dict(title='Z-score'),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1.0,
                xanchor='left',
                x=1.02,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1,
            ),
            hovermode='x unified',
            template='plotly_white',
            height=800,
            margin=dict(t=120, b=80, r=180),
        )

        # Arrow key navigation JS
        arrow_key_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv) {
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                        var xaxis = plotDiv._fullLayout.xaxis2;
                        var currentRange = xaxis.range;
                        var rangeWidth = currentRange[1] - currentRange[0];
                        var step = rangeWidth * 0.1;
                        
                        var newRange;
                        if (e.key === 'ArrowLeft') {
                            newRange = [currentRange[0] - step, currentRange[1] - step];
                        } else {
                            newRange = [currentRange[0] + step, currentRange[1] + step];
                        }
                        
                        if (newRange[0] < 0) {
                            newRange = [0, rangeWidth];
                        }
                        
                        Plotly.relayout(plotDiv, {'xaxis2.range': newRange});
                        e.preventDefault();
                    }
                });
            }
        });
        </script>
        """

        # Save HTML
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-multimodal_timeseries.html"
        )
        filepath = output_path / filename

        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html_content = html_content.replace('</body>', f'{arrow_key_js}</body>')

        with open(filepath, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        logger.info(f"Multimodal time series plot saved to: {filepath}")
        return filepath

    except Exception as exc:
        logger.error(f"Failed to generate multimodal time series plot: {exc}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_hrf_by_condition_4roi(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
    good_channels: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Generate HRF plots for 4 specific ROIs (Left/Right Anterior, Left/Right Posterior).

    ROIs defined as:
    Left Anterior: S1(AFF5h), S9(F7) -> AF7, F5, F7-F5, F7-AF7 (using S1 or S9 as source)
    Right Anterior: S2(AFF6h), S10(F8) -> AF8, F6, F8-F6, F8-AF8 (using S2 or S10 as source)
    Left Posterior: S3(CCP5h), S4(TTP7h), S5(TPP7h) (using S3, S4, S5 as source)
    Right Posterior: S6(CCP6h), S7(TTP8h), S8(TPP8h) (using S6, S7, S8 as source)

    Args:
        epochs: fNIRS epochs.
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_channels: List of good channel pairs (e.g., ["S1_D1"]) to include.
                       If None, uses all channels.

    Returns:
        Path to saved PNG file, or None on error.
    """
    logger.info("Generating fNIRS HRF by condition (4 ROIs)...")
    try:
        # Define ROIs by Source labels
        # We look for channels that have these sources in their label
        # Channel names are typically "S1_D1 hbo", etc.

        rois = {
            'Left Anterior': ['S1_', 'S9_'],
            'Right Anterior': ['S2_', 'S10_'],
            'Left Posterior': ['S3_', 'S4_', 'S5_'],
            'Right Posterior': ['S6_', 'S7_', 'S8_']
        }

        conditions = ['LEFT', 'RIGHT', 'NOTHING']
        colors = {'LEFT': 'green', 'RIGHT': 'purple', 'NOTHING': 'gray'}

        fig, axes = plt.subplots(4, 2, figsize=(15, 20))

        times = epochs.times

        for i, (roi_name, source_prefixes) in enumerate(rois.items()):
            # Find indices for this ROI, filtering by good_channels
            roi_indices_hbo = []
            roi_indices_hbr = []

            for ch_idx, ch_name in enumerate(epochs.ch_names):
                # Check if channel matches ROI source prefixes
                matches_source = any(prefix in ch_name for prefix in source_prefixes)

                # Check if channel is in good_channels list
                is_good = True
                if good_channels is not None:
                    # ch_name is like "S1_D1 hbo"
                    # good_channels has "S1_D1"
                    base_name = ch_name.split(' ')[0]
                    if base_name not in good_channels:
                        is_good = False

                if matches_source and is_good:
                    if 'hbo' in ch_name:
                        roi_indices_hbo.append(ch_idx)
                    elif 'hbr' in ch_name:
                        roi_indices_hbr.append(ch_idx)

            # Subplots for this ROI
            ax_hbo = axes[i, 0]
            ax_hbr = axes[i, 1]

            # Handle edge case: all ROI channels are bad (no good channels)
            if len(roi_indices_hbo) == 0 and len(roi_indices_hbr) == 0:
                # Display empty subplot with annotation (Requirement 5.5)
                ax_hbo.set_title(f"{roi_name} - HbO")
                ax_hbo.text(
                    0.5, 0.5, "No good channels",
                    ha='center', va='center',
                    transform=ax_hbo.transAxes,
                    fontsize=12, color='red', style='italic'
                )
                ax_hbo.set_ylabel('Concentration ($\\mu$M)')
                ax_hbo.grid(True, alpha=0.3)

                ax_hbr.set_title(f"{roi_name} - HbR")
                ax_hbr.text(
                    0.5, 0.5, "No good channels",
                    ha='center', va='center',
                    transform=ax_hbr.transAxes,
                    fontsize=12, color='red', style='italic'
                )
                ax_hbr.grid(True, alpha=0.3)

                logger.warning(f"ROI '{roi_name}' has no good channels - showing empty subplot")
                continue

            # Plot each condition
            for condition in conditions:
                if condition in epochs.event_id:
                    if len(roi_indices_hbo) > 0:
                        evoked = epochs[condition].average(picks=roi_indices_hbo)
                        if evoked.nave > 0:
                            mean_data = evoked.data.mean(axis=0) * 1e6
                            ax_hbo.plot(times, mean_data, label=f"{condition} (n={evoked.nave})", color=colors.get(condition, 'k'))

                    if len(roi_indices_hbr) > 0:
                        evoked = epochs[condition].average(picks=roi_indices_hbr)
                        if evoked.nave > 0:
                            mean_data = evoked.data.mean(axis=0) * 1e6
                            ax_hbr.plot(times, mean_data, label=f"{condition} (n={evoked.nave})", color=colors.get(condition, 'k'))

            # Styling
            ax_hbo.set_title(f"{roi_name} - HbO")
            ax_hbo.set_ylabel('Concentration ($\\mu$M)')
            ax_hbo.grid(True, alpha=0.3)
            ax_hbo.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            if i == 0: ax_hbo.legend()

            ax_hbr.set_title(f"{roi_name} - HbR")
            ax_hbr.grid(True, alpha=0.3)
            ax_hbr.axvline(x=0, color='k', linestyle='--', alpha=0.5)

        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 1].set_xlabel('Time (s)')

        plt.tight_layout()

        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_hrf_4roi.png"
        )
        filepath = output_path / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"fNIRS 4-ROI HRF plot saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to generate fNIRS 4-ROI HRF plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_block_average_mov_nomov(
    fnirs_epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
    good_channels: Optional[list[str]] = None,
) -> Optional[Path]:
    """Generate block-averaged HRF comparing contralateral MOV vs NO MOV.

    Creates a grid of subplots showing HbO time courses for each channel,
    with two conditions using CONTRALATERAL activation:
    - MOV (green): Contralateral motor trials only
        - Left hemisphere channels (S1_, S3_, S4_, S5_, S9_) → RIGHT hand epochs
        - Right hemisphere channels (S2_, S6_, S7_, S8_, S10_) → LEFT hand epochs
    - NO MOV (gray): NOTHING/rest condition

    Scientific rationale:
        Motor cortex shows contralateral activation: left motor cortex activates
        during right hand movement and vice versa. Using contralateral epochs
        provides cleaner motor-related HRF without ipsilateral contamination.

    Args:
        fnirs_epochs: MNE Epochs with fNIRS hemoglobin data containing
            LEFT, RIGHT, and NOTHING conditions.
        output_path: Directory to save the plot.
        config: SubjectConfig with subject info and task_duration.
        good_channels: Optional list of good channel pairs (e.g., ["S1_D1"]).
            If provided, only these channels are plotted.

    Returns:
        Path to saved PNG file, or None if failed.
    """
    logger.info("Generating fNIRS block average (contralateral MOV vs NO MOV)...")

    try:
        # Get HbO channels only
        hbo_channels = [ch for ch in fnirs_epochs.ch_names if "hbo" in ch.lower()]

        if not hbo_channels:
            logger.warning("No HbO channels found for block average")
            return None

        # Filter by good_channels if provided
        if good_channels is not None:
            hbo_channels = [
                ch for ch in hbo_channels
                if ch.split(" ")[0] in good_channels
            ]
            if not hbo_channels:
                logger.warning("No good HbO channels found for block average")
                return None

        n_channels = len(hbo_channels)
        times = fnirs_epochs.times

        # Define hemisphere membership based on source labels
        # Left hemisphere: S1, S3, S4, S5, S9 (and short S13, S15)
        # Right hemisphere: S2, S6, S7, S8, S10 (and short S14, S16)
        left_hemisphere_sources = ["S1_", "S3_", "S4_", "S5_", "S9_", "S13_", "S15_"]
        right_hemisphere_sources = ["S2_", "S6_", "S7_", "S8_", "S10_", "S14_", "S16_"]

        def get_channel_hemisphere(ch_name: str) -> str:
            """Determine hemisphere from channel name based on source label."""
            for src in left_hemisphere_sources:
                if src in ch_name:
                    return "left"
            for src in right_hemisphere_sources:
                if src in ch_name:
                    return "right"
            return "unknown"

        # Calculate grid size
        n_cols = min(4, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Get available conditions
        available_conditions = list(fnirs_epochs.event_id.keys())

        # Find LEFT and RIGHT conditions
        left_key = next((c for c in available_conditions if "LEFT" in c), None)
        right_key = next((c for c in available_conditions if "RIGHT" in c), None)
        nothing_key = next((c for c in available_conditions if "NOTHING" in c), None)

        # Colors for conditions
        mov_color = "#2ca02c"  # Green for MOV
        nomov_color = "#7f7f7f"  # Gray for NO MOV

        for idx, ch_name in enumerate(hbo_channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            ch_idx = fnirs_epochs.ch_names.index(ch_name)
            hemisphere = get_channel_hemisphere(ch_name)

            # Compute CONTRALATERAL MOV average
            # Left hemisphere channels → RIGHT hand epochs (contralateral)
            # Right hemisphere channels → LEFT hand epochs (contralateral)
            mov_data = None
            n_mov_trials = 0
            contralateral_label = ""

            if hemisphere == "left" and right_key is not None:
                # Left hemisphere → use RIGHT hand epochs
                right_epochs = fnirs_epochs[right_key]
                mov_data = right_epochs.get_data()[:, ch_idx, :]
                n_mov_trials = len(right_epochs)
                contralateral_label = "R"
            elif hemisphere == "right" and left_key is not None:
                # Right hemisphere → use LEFT hand epochs
                left_epochs = fnirs_epochs[left_key]
                mov_data = left_epochs.get_data()[:, ch_idx, :]
                n_mov_trials = len(left_epochs)
                contralateral_label = "L"
            elif hemisphere == "unknown":
                # Fallback: use both LEFT + RIGHT if hemisphere unknown
                mov_data_list = []
                if left_key is not None:
                    left_data = fnirs_epochs[left_key].get_data()[:, ch_idx, :]
                    mov_data_list.append(left_data)
                    n_mov_trials += len(fnirs_epochs[left_key])
                if right_key is not None:
                    right_data = fnirs_epochs[right_key].get_data()[:, ch_idx, :]
                    mov_data_list.append(right_data)
                    n_mov_trials += len(fnirs_epochs[right_key])
                if mov_data_list:
                    mov_data = np.concatenate(mov_data_list, axis=0)
                contralateral_label = "L+R"

            if mov_data is not None and len(mov_data) > 0:
                mov_mean = np.mean(mov_data, axis=0) * 1e6  # Convert to μM
                mov_sem = np.std(mov_data, axis=0) * 1e6 / np.sqrt(len(mov_data))

                ax.plot(times, mov_mean, color=mov_color, linewidth=1.5,
                        label=f"MOV-{contralateral_label} (n={n_mov_trials})")
                ax.fill_between(times, mov_mean - mov_sem, mov_mean + mov_sem,
                                color=mov_color, alpha=0.2)

            # Compute NO MOV average (NOTHING)
            if nothing_key is not None:
                nothing_epochs = fnirs_epochs[nothing_key]
                nothing_data = nothing_epochs.get_data()[:, ch_idx, :]
                n_nomov_trials = len(nothing_epochs)

                nomov_mean = np.mean(nothing_data, axis=0) * 1e6
                nomov_sem = np.std(nothing_data, axis=0) * 1e6 / np.sqrt(len(nothing_data))

                ax.plot(times, nomov_mean, color=nomov_color, linewidth=1.5,
                        label=f"NO MOV (n={n_nomov_trials})")
                ax.fill_between(times, nomov_mean - nomov_sem, nomov_mean + nomov_sem,
                                color=nomov_color, alpha=0.2)

            # Vertical lines
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline(config.trials.task_duration_sec, color="gray",
                       linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            # Channel label with hemisphere indicator
            short_name = ch_name.replace(" hbo", "").replace("_", "-")
            hemisphere_indicator = "L" if hemisphere == "left" else ("R" if hemisphere == "right" else "?")
            ax.set_title(f"{short_name} ({hemisphere_indicator})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("ΔHbO (μM)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7)

        # Hide empty subplots
        for idx in range(n_channels, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"fNIRS Block Average: Contralateral MOV vs NO MOV\n"
            f"Subject {config.subject.id} | Task duration: {config.trials.task_duration_sec}s\n"
            f"(L) = Left hemisphere → RIGHT hand | (R) = Right hemisphere → LEFT hand",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_block_average_mov_nomov.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"fNIRS block average (contralateral MOV vs NO MOV) saved to: {filepath}")
        return filepath

    except Exception as exc:
        logger.error(f"Failed to generate block average MOV vs NO MOV: {exc}")
        import traceback
        traceback.print_exc()
        return None


def load_sub012_montage(config: SubjectConfig) -> dict[str, Any]:
    """Load the extended 24-pair montage JSON for sub-012.

    Loads the montage file from the subject's BIDS session directory and
    validates that the required ChMontage key exists.

    Args:
        config: Subject configuration with data_root path.

    Returns:
        Parsed montage JSON containing ChMontage key.

    Raises:
        FileNotFoundError: If montage file does not exist.
        ValueError: If ChMontage key is missing.
    """
    montage_path = (
        config.data_root
        / f"sub-{config.subject.id}"
        / f"ses-{config.subject.session}"
        / "montage_combined_EEG_fNIRS_with_3Dcoords_approx.json"
    )

    if not montage_path.exists():
        logger.error(f"Montage file not found: {montage_path}")
        raise FileNotFoundError(f"Montage file not found: {montage_path}")

    with open(montage_path, "r") as montage_file:
        montage_json = json.load(montage_file)

    if "ChMontage" not in montage_json:
        logger.error(f"Montage file missing 'ChMontage' key: {montage_path}")
        raise ValueError(f"Montage file missing 'ChMontage' key: {montage_path}")

    logger.info(f"Loaded montage from: {montage_path}")
    return montage_json


def count_fnirs_channels(raw_fnirs: mne.io.Raw) -> int:
    """Count only fNIRS wavelength channels, excluding misc/AUX.

    Uses MNE's pick_types to select only fnirs_cw_amplitude channels,
    ensuring that auxiliary (misc) channels from the XDF stream are
    not included in the count.

    Args:
        raw_fnirs: MNE Raw object with fNIRS data.

    Returns:
        Number of fnirs_cw_amplitude channels.
    """
    fnirs_picks = mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[])
    return len(fnirs_picks)

def filter_channels_by_sci(
    raw_fnirs: mne.io.Raw,
    sci_threshold: float = 0.50,
) -> tuple[list[str], list[str]]:
    """Classify channels as good/bad based on SCI threshold.

    Calculates the Scalp Coupling Index (SCI) for all source-detector pairs
    and classifies them as good (SCI > threshold) or bad (SCI <= threshold).

    Args:
        raw_fnirs: MNE Raw object with fNIRS intensity data.
        sci_threshold: SCI threshold (channels with SCI <= threshold are bad).

    Returns:
        Tuple of (good_channel_pairs, bad_channel_pairs).
        Channel pairs are in format "S1_D1".

    Example:
        >>> good_pairs, bad_pairs = filter_channels_by_sci(raw_fnirs, sci_threshold=0.50)
        >>> print(f"Good: {len(good_pairs)}, Bad: {len(bad_pairs)}")
    """
    # Pick only fNIRS channels
    fnirs_picks = mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[])
    raw_fnirs_only = raw_fnirs.copy().pick(fnirs_picks)

    # Calculate SCI for all channels (threshold=0.0 to get all values)
    sci_values = calculate_sci(raw_fnirs_only, sci_threshold=0.0)

    good_pairs = [pair for pair, sci in sci_values.items() if sci > sci_threshold]
    bad_pairs = [pair for pair, sci in sci_values.items() if sci <= sci_threshold]

    logger.info(
        f"Channel filtering by SCI (threshold={sci_threshold}): "
        f"{len(good_pairs)} good, {len(bad_pairs)} bad"
    )

    return good_pairs, bad_pairs


def apply_fnirs_montage_from_json(
    raw: mne.io.Raw,
    montage_json: dict[str, Any],
) -> mne.io.Raw:
    """Apply 3D montage positions from JSON to fNIRS Raw object.

    Creates an MNE DigMontage from the source/detector 3D coordinates in the
    montage JSON and applies it to the Raw object. This enables proximity-based
    SCR pairing verification using actual spatial positions.

    For fNIRS data, MNE stores optode positions in the channel loc array:
    - loc[3:6] = source position (x, y, z)
    - loc[6:9] = detector position (x, y, z)
    
    MNE's source_detector_distances() computes distance as:
        np.linalg.norm(np.diff(loc[3:9].reshape(2, 3), axis=0))
    
    This is the distance between loc[3:6] and loc[6:9].

    Args:
        raw: MNE Raw object with fNIRS data.
        montage_json: Full montage JSON containing 'ChMontage' and 'OptodePositions'.

    Returns:
        Raw object with montage applied (modified in place, also returned).

    Notes:
        - Source positions are stored in loc[3:6]
        - Detector positions are stored in loc[6:9]
        - MNE uses loc[3:9] for distance calculation
    """
    ch_montage = montage_json.get("ChMontage", [])
    
    if not ch_montage:
        logger.warning("No ChMontage found in montage JSON, cannot apply 3D positions")
        return raw
    
    # Extract unique source and detector positions
    sources: dict[str, tuple[float, float, float]] = {}
    detectors: dict[str, tuple[float, float, float]] = {}
    
    for ch_info in ch_montage:
        # Extract source ID (e.g., "S1" from "S1_AFF5h")
        source_full = ch_info["source"]
        source_id = source_full.split("_")[0]  # "S1"
        
        # Extract detector ID (e.g., "D1" from "D1_AF7")
        detector_full = ch_info["detector"]
        detector_id = detector_full.split("_")[0]  # "D1"
        
        # Get 3D positions
        source_xyz = ch_info.get("source_xyz_m")
        detector_xyz = ch_info.get("detector_xyz_m")
        
        if source_xyz is not None and source_id not in sources:
            sources[source_id] = tuple(source_xyz)
        
        if detector_xyz is not None and detector_id not in detectors:
            detectors[detector_id] = tuple(detector_xyz)
    
    if not sources or not detectors:
        logger.warning("No source/detector positions found in montage JSON")
        return raw
    
    logger.info(
        f"Applying 3D montage: {len(sources)} sources, {len(detectors)} detectors"
    )
    
    # MNE fNIRS loc array layout (from MNE source code):
    # loc[3:6] = source position (x, y, z)
    # loc[6:9] = detector position (x, y, z)
    # loc[9] = wavelength (nm) - already set by build_fnirs_raw
    # 
    # MNE's source_detector_distances() uses:
    #   np.diff(loc[3:9].reshape(2, 3), axis=0)
    # which computes distance between loc[3:6] and loc[6:9]
    
    for ch_idx, ch_name in enumerate(raw.ch_names):
        # Parse channel name: "S1_D1 760" -> source="S1", detector="D1"
        parts = ch_name.split(" ")
        if len(parts) != 2:
            continue
        
        pair = parts[0]  # "S1_D1"
        pair_parts = pair.split("_")
        if len(pair_parts) != 2:
            continue
        
        source_id = pair_parts[0]  # "S1"
        detector_id = pair_parts[1]  # "D1"
        
        # Get positions
        source_pos = sources.get(source_id)
        detector_pos = detectors.get(detector_id)
        
        if source_pos is None or detector_pos is None:
            continue
        
        # Update channel loc array using MNE's expected layout:
        # loc[0:3] = channel position (midpoint for MNE-NIRS proximity matching)
        # loc[3:6] = source position (for MNE source_detector_distances)
        # loc[6:9] = detector position (for MNE source_detector_distances)
        
        # Compute midpoint for channel position
        midpoint = tuple((s + d) / 2.0 for s, d in zip(source_pos, detector_pos))
        
        raw.info["chs"][ch_idx]["loc"][0:3] = midpoint
        raw.info["chs"][ch_idx]["loc"][3:6] = source_pos
        raw.info["chs"][ch_idx]["loc"][6:9] = detector_pos
        
        # Compute and store actual source-detector distance
        distance = np.sqrt(
            (source_pos[0] - detector_pos[0])**2 +
            (source_pos[1] - detector_pos[1])**2 +
            (source_pos[2] - detector_pos[2])**2
        )
        raw.info["chs"][ch_idx]["loc"][10] = distance
    
    logger.info("3D positions applied to channel loc arrays")
    
    return raw


def mark_bad_channels_in_info(
    raw: mne.io.Raw,
    bad_pairs: list[str],
) -> None:
    """Mark all wavelength channels belonging to bad pairs as bads in MNE info.

    For each bad source-detector pair, marks both wavelength channels (760nm, 850nm)
    or both chromophore channels (hbo, hbr) as bad in the MNE info structure.
    This function modifies the Raw object in place.

    Args:
        raw: MNE Raw object (modified in place). Can also be an Epochs object.
        bad_pairs: List of bad channel pairs (e.g., ["S1_D1", "S2_D3"]).

    Example:
        >>> bad_pairs = ["S1_D1", "S3_D2"]
        >>> mark_bad_channels_in_info(raw_fnirs, bad_pairs)
        >>> print(raw_fnirs.info['bads'])  # Shows all wavelength channels for bad pairs
    """
    bad_channels = []
    for ch_name in raw.ch_names:
        # Channel names are "S1_D1 760" or "S1_D1 hbo"
        base_pair = ch_name.split(" ")[0]
        if base_pair in bad_pairs:
            bad_channels.append(ch_name)

    raw.info["bads"] = bad_channels
    logger.info(
        f"Marked {len(bad_channels)} channels as bad in MNE info "
        f"(from {len(bad_pairs)} bad pairs)"
    )

def verify_scr_pairing(
    raw_od: mne.io.Raw,
    short_channels: list[str],
    long_channels: list[str],
) -> dict[str, Any]:
    """Verify short channel regression pairing matches expected ROI mapping.

    Compares the MNE-NIRS proximity-based pairing (determined by 3D coordinates)
    against the expected anatomical mapping defined in SHORT_CHANNEL_ROI_MAP.
    Logs the actual pairing and identifies any mismatches.

    The function extracts channel positions from the MNE Raw object's loc array
    (populated by apply_fnirs_montage_from_json) and computes which long channels
    are spatially nearest to each short channel.

    Args:
        raw_od: MNE Raw object in optical density space. Must have valid 3D
            coordinates in the loc array for distance calculations.
        short_channels: List of short channel names (e.g., ["S13_D1", "S14_D3"]).
        long_channels: List of long channel names (e.g., ["S1_D1", "S2_D3"]).

    Returns:
        Dictionary with verification results:
            - pairing_correct: bool indicating if all pairings match expected
            - actual_pairing: dict mapping short channel → list of long channels
                (based on proximity from 3D coordinates)
            - expected_pairing: dict from SHORT_CHANNEL_ROI_MAP showing expected
                short → long channel mapping
            - mismatches: list of mismatch descriptions (empty if pairing_correct)

    Example:
        >>> result = verify_scr_pairing(raw_od, ["S13_D1"], ["S1_D1", "S9_D2"])
        >>> if not result["pairing_correct"]:
        ...     for mismatch in result["mismatches"]:
        ...         logger.warning(mismatch)
    """
    # Build expected pairing from SHORT_CHANNEL_ROI_MAP
    expected_pairing: dict[str, list[str]] = {}
    for short_ch, mapping in SHORT_CHANNEL_ROI_MAP.items():
        if short_ch in short_channels:
            # Find long channels that match the expected sources
            expected_long = []
            for long_ch in long_channels:
                source = long_ch.split("_")[0]
                if source in mapping["long_sources"]:
                    expected_long.append(long_ch)
            expected_pairing[short_ch] = sorted(expected_long)

    # Compute actual pairing based on 3D proximity
    actual_pairing: dict[str, list[str]] = {}

    # Get channel positions from the loc array
    # loc[0:3] = source position, loc[3:6] = detector position
    # Channel position = midpoint between source and detector
    ch_names = raw_od.ch_names
    
    def get_channel_position(ch_idx: int) -> Optional[np.ndarray]:
        """Get channel midpoint position from loc array."""
        loc = raw_od.info["chs"][ch_idx]["loc"]
        source_pos = loc[0:3]
        detector_pos = loc[3:6]
        
        # Check if positions are valid (not all zeros)
        if np.allclose(source_pos, 0) or np.allclose(detector_pos, 0):
            return None
        
        return (source_pos + detector_pos) / 2.0
    
    # Check if we have valid positions
    has_positions = False
    for ch_idx in range(len(ch_names)):
        if get_channel_position(ch_idx) is not None:
            has_positions = True
            break
    
    if not has_positions:
        logger.warning("No 3D positions found in loc arrays, cannot verify SCR pairing by proximity")
        return {
            "pairing_correct": False,
            "actual_pairing": {},
            "expected_pairing": expected_pairing,
            "mismatches": ["No 3D positions available for proximity-based verification"],
        }

    # For each short channel, find nearest long channels
    for short_ch in short_channels:
        # Get position of short channel (use first wavelength variant)
        short_ch_variants = [ch for ch in ch_names if ch.startswith(short_ch + " ")]
        if not short_ch_variants:
            logger.warning(f"Short channel {short_ch} not found in raw_od channels")
            continue

        short_ch_name = short_ch_variants[0]
        short_ch_idx = ch_names.index(short_ch_name)
        short_pos = get_channel_position(short_ch_idx)
        
        if short_pos is None:
            logger.warning(f"No position found for short channel {short_ch_name}")
            continue

        # Calculate distances to all long channels
        distances: list[tuple[str, float]] = []
        for long_ch in long_channels:
            long_ch_variants = [ch for ch in ch_names if ch.startswith(long_ch + " ")]
            if not long_ch_variants:
                continue

            long_ch_name = long_ch_variants[0]
            long_ch_idx = ch_names.index(long_ch_name)
            long_pos = get_channel_position(long_ch_idx)
            
            if long_pos is None:
                continue

            distance = np.linalg.norm(short_pos - long_pos)
            distances.append((long_ch, distance))

        # Sort by distance and assign nearest long channels
        # MNE-NIRS typically assigns based on proximity threshold or nearest neighbors
        # For verification, we check which long channels are closest to each short channel
        distances.sort(key=lambda x: x[1])

        # Get the expected number of long channels for this short channel
        expected_count = len(expected_pairing.get(short_ch, []))
        if expected_count > 0:
            nearest_long = [ch for ch, _ in distances[:expected_count]]
        else:
            # If no expected pairing, take channels within reasonable distance
            # Use a threshold based on typical ROI clustering (~50mm)
            nearest_long = [ch for ch, dist in distances if dist < 0.05]

        actual_pairing[short_ch] = sorted(nearest_long)

    # Compare actual vs expected and identify mismatches
    mismatches: list[str] = []
    for short_ch in short_channels:
        actual = set(actual_pairing.get(short_ch, []))
        expected = set(expected_pairing.get(short_ch, []))

        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            mismatch_desc = f"Short channel {short_ch}: "
            if missing:
                mismatch_desc += f"missing expected long channels {sorted(missing)}"
            if extra:
                if missing:
                    mismatch_desc += ", "
                mismatch_desc += f"unexpected long channels {sorted(extra)}"
            mismatches.append(mismatch_desc)

    pairing_correct = len(mismatches) == 0

    # Log the actual pairing
    logger.info("SCR Pairing Verification:")
    logger.info(f"  Pairing correct: {pairing_correct}")
    for short_ch, long_chs in actual_pairing.items():
        roi_name = SHORT_CHANNEL_ROI_MAP.get(short_ch, {}).get("roi", "Unknown")
        logger.info(f"  {short_ch} ({roi_name}) → {long_chs}")

    if mismatches:
        for mismatch in mismatches:
            logger.warning(f"  Mismatch: {mismatch}")

    return {
        "pairing_correct": pairing_correct,
        "actual_pairing": actual_pairing,
        "expected_pairing": expected_pairing,
        "mismatches": mismatches,
    }

def log_scr_noise_reduction(
    raw_od_before: mne.io.Raw,
    raw_od_after: mne.io.Raw,
    roi_map: dict[str, list[str]],
) -> dict[str, float]:
    """Log mean power reduction in systemic band (0.1-0.4 Hz) per ROI.

    Computes the Power Spectral Density (PSD) for channels in each ROI before
    and after Short Channel Regression (SCR), then calculates the percent
    power reduction in the systemic band (0.1-0.4 Hz) where Mayer waves and
    respiration artifacts are expected.

    The systemic band (0.1-0.4 Hz) captures:
    - Mayer waves (~0.1 Hz): Blood pressure oscillations
    - Respiration artifacts (~0.2-0.3 Hz): Breathing-related hemodynamics

    Args:
        raw_od_before: OD (optical density) data before SCR. Must contain
            fNIRS channels with valid data.
        raw_od_after: OD data after SCR. Must have the same channels as
            raw_od_before.
        roi_map: Mapping of ROI name to list of channel names (base pairs
            without wavelength suffix, e.g., ["S1_D1", "S1_D2"]).

    Returns:
        Dictionary mapping ROI name to percent power reduction (0-100 scale).
        Positive values indicate noise reduction; negative values indicate
        noise increase (unexpected).

    Example:
        >>> roi_map = {
        ...     "Left Anterior": ["S1_D1", "S9_D2"],
        ...     "Right Anterior": ["S2_D3", "S10_D4"],
        ... }
        >>> reductions = log_scr_noise_reduction(raw_od_before, raw_od_after, roi_map)
        >>> print(reductions)
        {'Left Anterior': 45.2, 'Right Anterior': 38.7, ...}
    """
    # Systemic band frequency range (Hz)
    systemic_fmin = 0.1
    systemic_fmax = 0.4

    power_reductions: dict[str, float] = {}

    logger.info("SCR Noise Reduction Analysis (systemic band 0.1-0.4 Hz):")

    for roi_name, channel_pairs in roi_map.items():
        # Find matching channels in the raw objects (both wavelengths)
        channels_before: list[str] = []
        channels_after: list[str] = []

        for ch_pair in channel_pairs:
            # Match channels that start with the pair name (e.g., "S1_D1 760", "S1_D1 850")
            matching_before = [
                ch for ch in raw_od_before.ch_names if ch.startswith(ch_pair + " ")
            ]
            matching_after = [
                ch for ch in raw_od_after.ch_names if ch.startswith(ch_pair + " ")
            ]
            channels_before.extend(matching_before)
            channels_after.extend(matching_after)

        if not channels_before or not channels_after:
            logger.warning(f"  {roi_name}: No matching channels found, skipping")
            power_reductions[roi_name] = 0.0
            continue

        # Pick only the channels that exist in both before and after
        common_channels = list(set(channels_before) & set(channels_after))
        if not common_channels:
            logger.warning(f"  {roi_name}: No common channels between before/after, skipping")
            power_reductions[roi_name] = 0.0
            continue

        try:
            # Compute PSD for before SCR
            raw_before_roi = raw_od_before.copy().pick(common_channels)
            psd_before = raw_before_roi.compute_psd(
                method="welch",
                fmin=systemic_fmin,
                fmax=systemic_fmax,
                n_fft=256,
                n_overlap=128,
                verbose=False,
            )
            # Get mean power across channels and frequencies
            power_before = psd_before.get_data().mean()

            # Compute PSD for after SCR
            raw_after_roi = raw_od_after.copy().pick(common_channels)
            psd_after = raw_after_roi.compute_psd(
                method="welch",
                fmin=systemic_fmin,
                fmax=systemic_fmax,
                n_fft=256,
                n_overlap=128,
                verbose=False,
            )
            power_after = psd_after.get_data().mean()

            # Calculate percent reduction
            if power_before > 0:
                percent_reduction = ((power_before - power_after) / power_before) * 100.0
            else:
                percent_reduction = 0.0

            power_reductions[roi_name] = percent_reduction

            logger.info(
                f"  {roi_name}: {percent_reduction:.1f}% power reduction "
                f"(before: {power_before:.2e}, after: {power_after:.2e})"
            )

        except Exception as exc:
            logger.warning(f"  {roi_name}: PSD computation failed - {exc}")
            power_reductions[roi_name] = 0.0

    # Summary
    valid_reductions = [v for v in power_reductions.values() if v != 0.0]
    if valid_reductions:
        mean_reduction = np.mean(valid_reductions)
        logger.info(f"  Mean power reduction across ROIs: {mean_reduction:.1f}%")
        if mean_reduction > 0:
            logger.info("  SCR successfully attenuated systemic noise (Mayer waves, respiration)")
        else:
            logger.warning("  SCR did not reduce systemic noise - check short channel pairing")

    return power_reductions


def apply_scr_with_explicit_pairing(
    raw_od: mne.io.Raw,
    roi_pairing: dict[str, list[str]],
) -> mne.io.Raw:
    """Apply short channel regression with explicit ROI-based pairing.

    This function implements a fallback SCR approach when MNE-NIRS proximity-based
    pairing produces incorrect assignments. Instead of relying on 3D coordinates,
    it uses the explicit ROI mapping defined in SHORT_CHANNEL_ROI_MAP.

    The GLM-based regression is applied per wavelength:
    - For each long channel, the corresponding short channel (from the same ROI)
      is used as the regressor
    - The regression removes superficial physiological noise (scalp blood flow,
      Mayer waves, respiration artifacts) while preserving cortical signals

    Algorithm:
    1. For each ROI, identify the short channel and its associated long channels
    2. For each wavelength (760nm, 850nm):
       - Build design matrix: X = [short_channel_signal, intercept]
       - Fit GLM: long_signal = β₀ + β₁ * short_signal + ε
       - Compute residuals: cleaned_long = long_signal - β₁ * short_signal

    Args:
        raw_od: MNE Raw object with optical density data (fnirs_od channels).
            Must have valid channel names in format "S{n}_D{m} {wavelength}".
        roi_pairing: Dictionary mapping short channel base name (e.g., "S13_D1")
            to list of long channel base names (e.g., ["S1_D1", "S9_D2"]).
            This overrides proximity-based pairing.

    Returns:
        MNE Raw object with regressed long channels (still in OD space).
        Short channels remain unchanged for reference.

    Raises:
        ValueError: If raw_od does not contain fnirs_od channel types.

    Example:
        >>> roi_pairing = {
        ...     "S13_D1": ["S1_D1", "S9_D2"],
        ...     "S14_D3": ["S2_D3", "S10_D4"],
        ... }
        >>> raw_regressed = apply_scr_with_explicit_pairing(raw_od, roi_pairing)

    References:
        - Saager & Berger (2005). Direct characterization of superficial
          contamination. J Biomed Opt 10(4).
        - Scholkmann et al. (2014). Review of fNIRS signal processing.

    Requirements: 8.5, 8.6
    """
    # Validate channel types
    channel_types = raw_od.get_channel_types()
    if not all(ch_type == "fnirs_od" for ch_type in channel_types):
        raise ValueError(
            "apply_scr_with_explicit_pairing() requires fnirs_od channel types. "
            f"Found: {set(channel_types)}"
        )

    logger.info("Applying SCR with explicit ROI-based pairing (fallback mode)")

    # Create a copy to avoid modifying original
    raw_result = raw_od.copy()

    # Get wavelengths from channel names (e.g., "760", "850")
    wavelengths = set()
    for ch_name in raw_od.ch_names:
        parts = ch_name.split(" ")
        if len(parts) >= 2:
            wavelengths.add(parts[-1])

    logger.info(f"  Wavelengths detected: {sorted(wavelengths)}")

    channels_regressed = 0
    rois_processed = 0
    rois_skipped: list[str] = []

    for short_ch_base, long_ch_bases in roi_pairing.items():
        roi_name = SHORT_CHANNEL_ROI_MAP.get(short_ch_base, {}).get("roi", "Unknown")
        logger.info(f"  Processing ROI: {roi_name} (short: {short_ch_base})")

        # Check if short channel is missing or bad for ALL wavelengths (Requirement 8.6)
        # This determines if the entire ROI lacks SCR capability
        short_ch_available_wavelengths: list[str] = []
        short_ch_missing_wavelengths: list[str] = []
        short_ch_bad_wavelengths: list[str] = []

        for wavelength in wavelengths:
            short_ch_name = f"{short_ch_base} {wavelength}"
            if short_ch_name not in raw_od.ch_names:
                short_ch_missing_wavelengths.append(wavelength)
            elif short_ch_name in raw_od.info["bads"]:
                short_ch_bad_wavelengths.append(wavelength)
            else:
                short_ch_available_wavelengths.append(wavelength)

        # If short channel is unavailable for ALL wavelengths, skip entire ROI
        if not short_ch_available_wavelengths:
            if short_ch_missing_wavelengths and short_ch_bad_wavelengths:
                reason = "missing from montage and marked as bad"
            elif short_ch_missing_wavelengths:
                reason = "missing from montage"
            else:
                reason = "marked as Bad_Channel"

            logger.warning(
                f"  ROI '{roi_name}' lacks short channel regression: "
                f"short channel {short_ch_base} is {reason}. "
                f"Proceeding without SCR for this ROI's long channels."
            )
            rois_skipped.append(roi_name)
            rois_processed += 1
            continue

        for wavelength in wavelengths:
            # Find the short channel for this wavelength
            short_ch_name = f"{short_ch_base} {wavelength}"
            if short_ch_name not in raw_od.ch_names:
                logger.warning(f"    Short channel {short_ch_name} not found, skipping wavelength")
                continue

            if short_ch_name in raw_od.info["bads"]:
                logger.warning(f"    Short channel {short_ch_name} is marked bad, skipping wavelength")
                continue

            # Get short channel data
            short_idx = raw_od.ch_names.index(short_ch_name)
            short_signal = raw_od._data[short_idx]

            # Regress each long channel
            for long_ch_base in long_ch_bases:
                long_ch_name = f"{long_ch_base} {wavelength}"
                if long_ch_name not in raw_od.ch_names:
                    continue

                if long_ch_name in raw_od.info["bads"]:
                    continue

                long_idx = raw_result.ch_names.index(long_ch_name)
                long_signal = raw_result._data[long_idx]

                # GLM regression: long = β₀ + β₁ * short + ε
                # Design matrix: [short_signal, ones]
                design_matrix = np.column_stack([short_signal, np.ones(len(short_signal))])

                # Solve least squares: β = (X'X)^(-1) X'y
                try:
                    beta, residuals, rank, singular_values = np.linalg.lstsq(
                        design_matrix, long_signal, rcond=None
                    )
                    beta_short = beta[0]

                    # Compute cleaned signal (remove short channel contribution)
                    # Keep intercept to preserve DC offset
                    cleaned_signal = long_signal - beta_short * short_signal

                    # Update the result
                    raw_result._data[long_idx] = cleaned_signal
                    channels_regressed += 1

                except np.linalg.LinAlgError as exc:
                    logger.warning(
                        f"    GLM failed for {long_ch_name}: {exc}, skipping"
                    )

        rois_processed += 1

    # Log summary including skipped ROIs (Requirement 8.6)
    logger.info(
        f"  Explicit SCR complete: {channels_regressed} channels regressed "
        f"across {rois_processed} ROIs"
    )
    if rois_skipped:
        logger.warning(
            f"  ROIs without SCR (missing/bad short channels): {rois_skipped}"
        )

    return raw_result


def process_fnirs_with_scr_fallback(
    raw_fnirs: mne.io.Raw,
    montage_config: list[dict[str, Any]],
    montage_json: dict[str, Any],
    config: SubjectConfig,
    external_baseline: Optional[ExternalBaselineResult] = None,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Process fNIRS data with SCR verification and ROI-based fallback.

    This function implements the fNIRS processing pipeline with an additional
    verification step for Short Channel Regression (SCR). If the MNE-NIRS
    proximity-based pairing produces incorrect assignments (short channels
    paired with long channels from different ROIs), it falls back to explicit
    ROI-based pairing.

    Processing order (following MNE-NIRS best practices):
    0. (Optional) Apply external baseline subtraction on intensity
    1. Intensity → Optical Density (OD)
    2. Motion correction (TDDR on OD)
    3. Identify short/long channels
    3b. Apply 3D montage from JSON for proximity verification
    4. Verify SCR pairing against expected ROI mapping
    5. Apply SCR (proximity-based or fallback to explicit ROI-based)
    6. OD → Hemoglobin (Beer-Lambert)
    7. Bandpass filter (0.01-0.5 Hz on Hb)

    Args:
        raw_fnirs: MNE Raw object with raw fNIRS intensity data.
        montage_config: Channel montage configuration list from JSON (ChMontage).
        montage_json: Full montage JSON with 3D coordinates for SCR verification.
        config: Subject configuration with processing parameters.
        external_baseline: Optional ExternalBaselineResult from a separate baseline
            recording. If provided, baseline subtraction is applied to intensity
            data before OD conversion.

    Returns:
        Tuple of (processed_fnirs, processing_metrics):
            - processed_fnirs: MNE Raw with filtered hemoglobin data
            - processing_metrics: Dictionary with processing statistics

    Requirements: 8.5
    """
    logger.info("=" * 80)
    logger.info("fNIRS Processing with SCR Verification and Fallback")
    logger.info("=" * 80)

    processing_metrics: dict[str, Any] = {
        "scr_pairing_correct": None,
        "scr_fallback_used": False,
        "scr_mismatches": [],
        "processing_steps": [],
        "external_baseline_applied": external_baseline is not None,
    }

    # Filter to only fNIRS channels (exclude AUX/misc)
    fnirs_picks = mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[])
    raw_intensity = raw_fnirs.copy().pick(fnirs_picks)

    # Step 0: Apply 3D montage FIRST (before any processing)
    # This ensures correct distances are preserved through OD conversion and motion correction
    logger.info("Step 0: Applying 3D montage from JSON (before processing)")
    raw_intensity = apply_fnirs_montage_from_json(raw_intensity, montage_json)
    processing_metrics["processing_steps"].append("montage_3d_applied")

    # Step 0b: Apply external baseline normalization (I / I₀) - state-of-the-art approach
    if external_baseline is not None:
        logger.info(
            f"Step 0b: Applying external baseline normalization (I/I₀) from {external_baseline.source_file.name} "
            f"({external_baseline.duration_sec:.1f}s, {external_baseline.n_channels} channels)"
        )
        raw_intensity = apply_external_baseline_to_fnirs(raw_intensity, external_baseline)
        processing_metrics["processing_steps"].append("external_baseline_normalization")
        processing_metrics["external_baseline_duration_sec"] = external_baseline.duration_sec
        processing_metrics["external_baseline_source"] = str(external_baseline.source_file)

    # Step 1: Intensity → Optical Density
    logger.info("Step 1: Converting intensity to optical density")
    raw_od = convert_to_optical_density(raw_intensity)
    processing_metrics["processing_steps"].append("optical_density_conversion")

    # Step 2: Motion correction (TDDR)
    logger.info("Step 2: Applying motion correction (TDDR)")
    raw_od_corrected = correct_motion_artifacts(raw_od, method="tddr")
    processing_metrics["processing_steps"].append("motion_correction_tddr")

    # Step 2b: Re-apply 3D montage AFTER OD conversion and motion correction
    # MNE's optical_density() and TDDR may not preserve loc array positions
    # This ensures correct distances for MNE-NIRS short channel detection
    logger.info("Step 2b: Re-applying 3D montage (positions lost during OD/TDDR)")
    raw_od_corrected = apply_fnirs_montage_from_json(raw_od_corrected, montage_json)
    processing_metrics["processing_steps"].append("montage_3d_reapplied")

    # Step 3: Identify short and long channels
    logger.info("Step 3: Identifying short and long channels")
    short_channels, long_channels = identify_short_channels(
        raw_od_corrected,
        montage_config,
        short_threshold_mm=config.quality.short_channel_distance_mm,
    )
    logger.info(f"  Found {len(short_channels)} short channels: {short_channels}")
    logger.info(f"  Found {len(long_channels)} long channels")

    # Step 4: Verify SCR pairing
    logger.info("Step 4: Verifying SCR pairing against expected ROI mapping")
    scr_verification = verify_scr_pairing(raw_od_corrected, short_channels, long_channels)
    processing_metrics["scr_pairing_correct"] = scr_verification["pairing_correct"]
    processing_metrics["scr_mismatches"] = scr_verification["mismatches"]

    # Step 5: Apply SCR (with fallback if needed)
    raw_od_before_scr = raw_od_corrected.copy()

    if scr_verification["pairing_correct"]:
        logger.info("Step 5: SCR pairing correct, using proximity-based regression")
        raw_od_regressed = apply_short_channel_regression(
            raw_od_corrected, short_channels, long_channels
        )
        processing_metrics["processing_steps"].append("scr_proximity_based")
    else:
        # Log warning about mismatches (Requirement 8.5)
        logger.warning("=" * 60)
        logger.warning("SCR PAIRING MISMATCH DETECTED - FALLING BACK TO ROI-BASED")
        logger.warning("=" * 60)
        for mismatch in scr_verification["mismatches"]:
            logger.warning(f"  {mismatch}")

        # Build explicit ROI pairing from SHORT_CHANNEL_ROI_MAP
        # Also check for missing short channels (Requirement 8.6)
        roi_pairing: dict[str, list[str]] = {}
        missing_short_channels: list[tuple[str, str]] = []
        
        for short_ch, mapping in SHORT_CHANNEL_ROI_MAP.items():
            if short_ch in short_channels:
                # Find long channels matching the expected sources
                expected_long = []
                for long_ch in long_channels:
                    source = long_ch.split("_")[0]
                    if source in mapping["long_sources"]:
                        expected_long.append(long_ch)
                roi_pairing[short_ch] = expected_long
            else:
                # Short channel missing from montage (Requirement 8.6)
                missing_short_channels.append((short_ch, mapping["roi"]))
        
        # Log warnings for missing short channels
        for short_ch, roi_name in missing_short_channels:
            logger.warning(
                f"  ROI '{roi_name}' lacks short channel regression: "
                f"short channel {short_ch} is missing from montage. "
                f"Proceeding without SCR for this ROI's long channels."
            )

        logger.info("Step 5: Applying SCR with explicit ROI-based pairing (fallback)")
        raw_od_regressed = apply_scr_with_explicit_pairing(raw_od_corrected, roi_pairing)
        processing_metrics["scr_fallback_used"] = True
        processing_metrics["processing_steps"].append("scr_roi_based_fallback")

    # Log noise reduction per ROI
    roi_channel_map = {
        "Left Anterior": [lc for lc in long_channels if lc.split("_")[0] in ["S1", "S9"]],
        "Right Anterior": [lc for lc in long_channels if lc.split("_")[0] in ["S2", "S10"]],
        "Left Posterior": [lc for lc in long_channels if lc.split("_")[0] in ["S3", "S4", "S5"]],
        "Right Posterior": [lc for lc in long_channels if lc.split("_")[0] in ["S6", "S7", "S8"]],
    }
    noise_reductions = log_scr_noise_reduction(raw_od_before_scr, raw_od_regressed, roi_channel_map)
    processing_metrics["noise_reduction_by_roi"] = noise_reductions

    # Step 6: OD → Hemoglobin (Beer-Lambert)
    logger.info("Step 6: Converting optical density to hemoglobin")
    raw_haemo = convert_to_hemoglobin(raw_od_regressed, dpf=6.0)
    processing_metrics["processing_steps"].append("hemoglobin_conversion")

    # Step 7: Bandpass filter
    logger.info("Step 7: Filtering hemoglobin data (0.01-0.5 Hz)")
    raw_haemo_filtered = filter_hemoglobin_data(
        raw_haemo,
        l_freq=config.filters.fnirs_bandpass_low_hz,
        h_freq=config.filters.fnirs_bandpass_high_hz,
    )
    processing_metrics["processing_steps"].append("bandpass_filter")

    logger.info("=" * 80)
    logger.info("fNIRS processing with SCR fallback complete")
    logger.info(f"  SCR pairing correct: {processing_metrics['scr_pairing_correct']}")
    logger.info(f"  Fallback used: {processing_metrics['scr_fallback_used']}")
    logger.info("=" * 80)

    return raw_haemo_filtered, processing_metrics


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Determine and log modality mode
    modality_mode = determine_modality_mode(args)
    logger.info(f"Pipeline modality mode: {modality_mode.name}")

    # 1. Load Configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
        
    config = SubjectConfig.from_yaml(args.config)
    main_pipeline.print_configuration_summary(config)
    
    # 2. Custom Data Loading (Handle sub-12 vs sub-012)
    # Explicitly check for the 'sub-12' filename variant
    xdf_filename = f"sub-12_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
    xdf_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / xdf_filename
    
    if not xdf_path.exists():
        # Try flat structure
        xdf_path = config.data_root / f"sub-{config.subject.id}" / xdf_filename
        
    # If still not found, try the standard 'sub-012' name using the main pipeline's logic
    if not xdf_path.exists():
        logger.warning(f"Custom sub-12 file not found at {xdf_path}, trying standard naming...")
        try:
             # This might fail if load_and_identify_streams is too strict, but worth a try as fallback
             streams = main_pipeline.load_and_identify_streams(config)
        except Exception:
             logger.error(f"XDF file not found. Checked custom path: {xdf_path}")
             sys.exit(1)
    else:
        logger.info(f"Loading XDF from custom path: {xdf_path}")
        streams, header = load_xdf_file(xdf_path)
        streams = identify_streams(streams)

    # 3. Build MNE Object (Custom for Sub-012)
    # Load montage using dedicated function with error handling
    try:
        montage_json = load_sub012_montage(config)
    except (FileNotFoundError, ValueError) as error:
        logger.error(f"Failed to load montage: {error}")
        sys.exit(1)
        
    # Build objects manually to bypass main pipeline's JSON lookup
    raw_eeg = None
    raw_fnirs = None
    
    eeg_stream = streams.get("eeg")
    fnirs_stream = streams.get("fnirs")
    marker_stream = streams.get("markers")  # Note: "markers" with 's'
    
    # Build EEG Raw object (skip if FNIRS_ONLY mode)
    if modality_mode != ModalityMode.FNIRS_ONLY:
        if eeg_stream:
            logger.info("Building EEG Raw object...")
            eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(eeg_stream)
            raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, eeg_stream['info'], eeg_timestamps)
            logger.info(f"marker_stream is None: {marker_stream is None}")
            if marker_stream:
                logger.info("Calling embed_events...")
                raw_eeg = embed_events(raw_eeg, marker_stream)
                logger.info(f"EEG annotations after embed_events: {set(raw_eeg.annotations.description)}")
            else:
                logger.warning("No marker stream found, skipping embed_events")
        
        if raw_eeg is None:
            logger.error("No EEG stream found")
            sys.exit(1)
    else:
        logger.info("Skipping EEG loading (--fnirs-only mode)")
            
    # Build fNIRS Raw object (skip if EEG_ONLY mode)
    if modality_mode != ModalityMode.EEG_ONLY:
        if fnirs_stream:
            logger.info("Building fNIRS Raw object with custom montage...")
            fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
            # Use our loaded montage_json
            raw_fnirs = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_json['ChMontage'], fnirs_timestamps)
            if marker_stream:
                raw_fnirs = embed_events(raw_fnirs, marker_stream)

        if raw_fnirs is not None:
            logger.info(f"fNIRS stream found: {len(raw_fnirs.ch_names)} channels (including misc)")
        else:
            logger.warning("No fNIRS stream found")
    else:
        logger.info("Skipping fNIRS loading (--eeg-only mode)")
        
    # 4. Synthesize NOTHING conditions
    # Task is 7s, NOTHING epochs are 6s (matching LEFT/RIGHT epoch duration)
    eeg_synthesis_stats = None
    fnirs_synthesis_stats = None
    
    if modality_mode != ModalityMode.FNIRS_ONLY and raw_eeg is not None:
        # NOTHING epochs: 7s task window (matching LEFT/RIGHT)
        # rest_duration_cap=7.0 ensures we have enough rest for the full epoch
        raw_eeg, eeg_synthesis_stats = synthesize_nothing_annotations(
            raw_eeg, task_duration=7.0, rest_duration_cap=7.0
        )
    
    if modality_mode != ModalityMode.EEG_ONLY and raw_fnirs is not None:
        # Run synthesis on fNIRS too — it has its own copy of annotations
        raw_fnirs, fnirs_synthesis_stats = synthesize_nothing_annotations(
            raw_fnirs, task_duration=7.0, rest_duration_cap=7.0
        )

    # Log available annotations after synthesis
    if raw_eeg is not None:
        logger.info(f"EEG annotations after synthesis: {set(raw_eeg.annotations.description)}")
    if raw_fnirs is not None:
        logger.info(f"fNIRS annotations after synthesis: {set(raw_fnirs.annotations.description)}")
    
    # 5. Preprocessing (Reuse main pipeline)
    # Output path
    output_path = config.output_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize viz_paths
    viz_paths = {}
    
    # OVERRIDE SCI THRESHOLD (Task 4)
    # QualityThresholds is frozen, so we create a new instance with the modified value
    from dataclasses import replace
    original_sci = config.quality.sci_threshold
    new_quality = replace(config.quality, sci_threshold=0.50)
    config.quality = new_quality
    logger.info(f"Overriding SCI threshold from {original_sci} to {config.quality.sci_threshold}")

    # Collect good channels list (Task 4)
    good_fnirs_channels = None

    # NEW: Generate SCI Comparison Plot (Initial vs Final)
    # Must be done on raw intensity data (before preprocessing/OD conversion)
    # Skip if EEG_ONLY mode
    if modality_mode != ModalityMode.EEG_ONLY and raw_fnirs is not None:
        # Pick only fNIRS channels for SCI calculation to avoid 'misc'
        fnirs_picks = mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[])
        raw_fnirs_only = raw_fnirs.copy().pick(fnirs_picks)
        
        sci_plot_path = generate_sci_comparison_plot(raw_fnirs_only, output_path, config)
        if sci_plot_path:
            viz_paths['fnirs_sci_comparison'] = sci_plot_path
            
        # Determine good/bad channels based on SCI threshold (0.50)
        # Uses filter_channels_by_sci() to properly classify channels
        good_fnirs_channels, bad_fnirs_channels = filter_channels_by_sci(
            raw_fnirs, sci_threshold=config.quality.sci_threshold
        )

    # Pass both EEG and fNIRS (or None based on modality mode)
    eeg_for_preprocessing = raw_eeg if modality_mode != ModalityMode.FNIRS_ONLY else None
    
    # For fNIRS, use custom preprocessing with SCR verification and fallback
    processed_fnirs = None
    fnirs_processing_metrics = None
    
    if modality_mode != ModalityMode.EEG_ONLY and raw_fnirs is not None:
        # Load external baseline from separate baseline recording (60s stable baseline)
        # This provides more stable baseline than 1-second pre-stimulus windows
        external_baseline = None
        baseline_xdf_filename = f"sub-12_ses-S001_task-baseline_run-001_eeg.xdf"
        baseline_xdf_path = (
            config.data_root / f"sub-{config.subject.id}" / 
            f"ses-{config.subject.session}" / baseline_xdf_filename
        )
        
        if baseline_xdf_path.exists():
            try:
                logger.info(f"Loading external baseline from: {baseline_xdf_path}")
                external_baseline = load_external_fnirs_baseline(
                    baseline_xdf_path,
                    baseline_duration_sec=60.0,
                    marker_name="baseline",
                )
                logger.info(
                    f"External baseline loaded: {external_baseline.n_channels} channels, "
                    f"{external_baseline.duration_sec:.1f}s duration"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to load external baseline: {exc}. "
                    "Proceeding without external baseline correction."
                )
                external_baseline = None
        else:
            logger.info(
                f"External baseline file not found: {baseline_xdf_path}. "
                "Proceeding without external baseline correction."
            )
        
        logger.info("Running custom fNIRS preprocessing with SCR fallback...")
        processed_fnirs, fnirs_processing_metrics = process_fnirs_with_scr_fallback(
            raw_fnirs,
            montage_json["ChMontage"],
            montage_json,  # Pass full JSON for 3D coordinates
            config,
            external_baseline=external_baseline,
        )
        logger.info("Custom fNIRS preprocessing complete")
    
    # Run EEG preprocessing through main pipeline (if needed)
    processed_eeg = None
    if modality_mode != ModalityMode.FNIRS_ONLY and eeg_for_preprocessing is not None:
        logger.info("Running EEG preprocessing through main pipeline...")
        processed_eeg, _ = main_pipeline.run_preprocessing(
            raw_eeg=eeg_for_preprocessing,
            raw_fnirs=None,  # fNIRS already processed above
            config=config,
            output_path=output_path
        )
    
    if modality_mode != ModalityMode.FNIRS_ONLY and processed_eeg is None:
        logger.error("EEG Preprocessing failed")
        sys.exit(1)
        
    # 6. Analysis and Visualization
    # Run Standard EEG Analysis (Epoching, TFR, ERD/ERS) - skip if FNIRS_ONLY mode
    eeg_results = None
    if modality_mode != ModalityMode.FNIRS_ONLY:
        logger.info("Running EEG Analysis...")
        eeg_results = main_pipeline.run_eeg_analysis(processed_eeg, config, output_path)
    else:
        logger.info("Skipping EEG analysis (--fnirs-only mode)")
    
    # Run fNIRS Analysis - skip if EEG_ONLY mode
    fnirs_results = None
    if modality_mode != ModalityMode.EEG_ONLY and processed_fnirs is not None:
        logger.info("Running fNIRS Analysis...")
        # Check if function exists
        if hasattr(main_pipeline, 'run_fnirs_analysis'):
            fnirs_results = main_pipeline.run_fnirs_analysis(processed_fnirs, config)
        else:
            logger.warning("run_fnirs_analysis function not found in main_pipeline")
    elif modality_mode == ModalityMode.EEG_ONLY:
        logger.info("Skipping fNIRS analysis (--eeg-only mode)")

    # 6.5 Validate NOTHING condition integrity (advisory, non-blocking)
    logger.info("Validating NOTHING condition...")
    fnirs_epochs_for_validation = (
        fnirs_results["epochs"] if fnirs_results and "epochs" in fnirs_results else None
    )
    eeg_epochs_for_validation = (
        eeg_results["epochs"] if eeg_results and "epochs" in eeg_results else None
    )
    
    # Only validate if we have at least one modality's epochs
    if eeg_epochs_for_validation is not None or fnirs_epochs_for_validation is not None:
        validation_result = validate_nothing_condition(
            eeg_epochs=eeg_epochs_for_validation,
            fnirs_epochs=fnirs_epochs_for_validation,
        )
        if validation_result.all_passed:
            logger.info(
                "NOTHING validation: all checks passed "
                "(%d NOTHING, %d LEFT, %d RIGHT epochs)",
                validation_result.n_nothing_epochs,
                validation_result.n_left_epochs,
                validation_result.n_right_epochs,
            )
        else:
            logger.warning(
                "NOTHING validation: %d issue(s) detected — %s",
                len(validation_result.warnings),
                "; ".join(validation_result.warnings),
            )
    else:
        logger.info("Skipping NOTHING validation (no epochs available)")

    # Process EEG Results (CSP, Viz) - skip if FNIRS_ONLY mode
    # viz_paths is already initialized above
    if modality_mode != ModalityMode.FNIRS_ONLY and eeg_results:
        # Run CSP (LEFT vs RIGHT)
        # Note: This function requires 'epochs' in eeg_results
        logger.info("Running CSP Analysis (LEFT vs RIGHT)...")
        csp_path, csp_results = main_pipeline.generate_csp_analysis(eeg_results['epochs'], output_path, config)
        eeg_results['csp_analysis_path'] = csp_path
        eeg_results['csp_results'] = csp_results
        
        # Run CSP (MOV vs REST)
        # Verify module path and function existence
        logger.info(f"Loaded main_pipeline from: {main_pipeline.__file__}")
        if not hasattr(main_pipeline, 'generate_csp_movement_vs_rest'):
             logger.error("generate_csp_movement_vs_rest NOT FOUND in main_pipeline!")
        
        logger.info(f"Event IDs in epochs: {eeg_results['epochs'].event_id}")

        logger.info("Running CSP Analysis (MOV vs NO MOV)...")
        csp_mov_path, csp_mov_results = main_pipeline.generate_csp_movement_vs_rest(eeg_results['epochs'], output_path, config)
        eeg_results['csp_mov_vs_rest_path'] = csp_mov_path
        eeg_results['csp_mov_results'] = csp_mov_results
        
        # Generate Individual Visualizations with all 3 conditions (LEFT, RIGHT, NOTHING)
        logger.info("Generating individual visualizations with all conditions...")
        epochs = eeg_results['epochs']
        
        # Add CSP paths to viz_paths for report generation
        if 'csp_analysis_path' in eeg_results and eeg_results['csp_analysis_path']:
             viz_paths['eeg_csp_analysis'] = eeg_results['csp_analysis_path']
        if 'csp_mov_vs_rest_path' in eeg_results and eeg_results['csp_mov_vs_rest_path']:
             viz_paths['eeg_csp_mov_vs_rest'] = eeg_results['csp_mov_vs_rest_path']

        # TFR Maps (shows LEFT, RIGHT, NOTHING in separate columns)
        logger.info("Generating TFR maps...")
        tfr_path = main_pipeline.generate_tfr_maps(epochs, output_path, config)
        if tfr_path:
            viz_paths['eeg_tfr_maps'] = tfr_path
        
        # ERP Analysis (shows all 3 conditions)
        logger.info("Generating ERP analysis...")
        erp_path = main_pipeline.generate_erp_analysis(epochs, output_path, config)
        if erp_path:
            viz_paths['eeg_erp_analysis'] = erp_path
        
        # Clustered TFR Maps
        logger.info("Generating clustered TFR maps...")
        clustered_tfr_path = main_pipeline.generate_clustered_tfr_maps(epochs, output_path, config)
        if clustered_tfr_path:
            viz_paths['eeg_tfr_maps_roi'] = clustered_tfr_path
        
        # Beta Topoplots
        logger.info("Generating beta topoplots...")
        beta_topo_path = main_pipeline.generate_beta_topoplots(epochs, output_path, config)
        if beta_topo_path:
            viz_paths['eeg_beta_topoplot'] = beta_topo_path
        
        # Contralateral ERD plots
        logger.info("Generating contralateral ERD plots...")
        contralat_timecourse, contralat_topo = main_pipeline.generate_contralateral_erd_plots(epochs, output_path, config)
        if contralat_topo:
            viz_paths['eeg_contralateral_topoplot'] = contralat_topo
        if contralat_timecourse:
            viz_paths['eeg_contralateral_timecourse'] = contralat_timecourse
            
        # Contrast Analysis
        logger.info("Generating contrast analysis...")
        contrast_path, lat_index_path = main_pipeline.generate_contrast_analysis(epochs, output_path, config)
        if contrast_path:
            viz_paths['eeg_contrast_analysis'] = contrast_path
    elif modality_mode == ModalityMode.FNIRS_ONLY:
        logger.info("Skipping EEG visualizations (--fnirs-only mode)")
    else:
        logger.error("EEG Analysis failed to produce results.")
        
    # fNIRS Visualizations - skip if EEG_ONLY mode
    if modality_mode != ModalityMode.EEG_ONLY and fnirs_results and 'epochs' in fnirs_results:
        logger.info("Generating fNIRS visualizations with all conditions...")
        fnirs_epochs = fnirs_results['epochs']
        
        # HRF by Condition (Original + 4 ROI)
        logger.info("Generating fNIRS HRF by condition...")
        # Original function (might filter internally, but let's trust it for now or replace call if needed)
        # The user asked to "Expand HRF by condition to 4 ROIs". 
        # We'll use our NEW function for the 4-ROI plot.
        
        hrf_4roi_path = generate_fnirs_hrf_by_condition_4roi(fnirs_epochs, output_path, config, good_channels=good_fnirs_channels)
        if hrf_4roi_path:
            viz_paths['fnirs_hrf_by_condition_4roi'] = hrf_4roi_path
            
        # Keep original as well if useful, or assume 4-ROI replaces it? 
        # Implementation plan item 6 says "Expand...". Let's run potentially both or just the new one.
        # But the user also asked for "Full-experiment HBO/HBR time series plot" (Task 5)
        
        # Time Series Plot (Task 5) - Global (all channels)
        if processed_fnirs is not None:
             ts_path = generate_fnirs_timeseries_plot(processed_fnirs, output_path, config, good_channels=good_fnirs_channels)
             if ts_path:
                 viz_paths['fnirs_timeseries'] = ts_path
             
             # Time Series Plots by ROI (4 additional plots)
             roi_definitions_for_timeseries = {
                 'Left Anterior': ['S1_', 'S9_'],
                 'Right Anterior': ['S2_', 'S10_'],
                 'Left Posterior': ['S3_', 'S4_', 'S5_'],
                 'Right Posterior': ['S6_', 'S7_', 'S8_'],
             }
             
             for roi_name, source_prefixes in roi_definitions_for_timeseries.items():
                 roi_ts_path = generate_fnirs_timeseries_plot_by_roi(
                     processed_fnirs,
                     output_path,
                     config,
                     roi_name=roi_name,
                     source_prefixes=source_prefixes,
                     good_channels=good_fnirs_channels,
                 )
                 if roi_ts_path:
                     roi_key = f"fnirs_timeseries_{roi_name.lower().replace(' ', '_')}"
                     viz_paths[roi_key] = roi_ts_path
             
             # All ROIs combined in one plot
             all_rois_ts_path = generate_fnirs_timeseries_all_rois(
                 processed_fnirs,
                 output_path,
                 config,
                 good_channels=good_fnirs_channels,
             )
             if all_rois_ts_path:
                 viz_paths['fnirs_timeseries_all_rois'] = all_rois_ts_path
             
             # Multimodal Time Series Plot (fNIRS + EEG alpha/beta)
             # Only available in FULL_MULTIMODAL mode (requires both modalities)
             if modality_mode == ModalityMode.FULL_MULTIMODAL and processed_eeg is not None:
                 multimodal_ts_path = generate_multimodal_timeseries_plot(
                     raw_haemo=processed_fnirs,
                     raw_eeg=processed_eeg,
                     output_path=output_path,
                     config=config,
                     good_fnirs_channels=good_fnirs_channels,
                     eeg_channels=['C3', 'C4', 'Cz', 'CP3', 'CP4'],
                 )
                 if multimodal_ts_path:
                     viz_paths['multimodal_timeseries'] = multimodal_ts_path
        
        # Block Average (Task 4: Filter bad channels)
        # Modifying main pipeline calls isn't easy without changing main pipeline code. 
        # But we can try to pass `picks` if the function supports it, or just rely on the new plots.
        # Task 4 said "Modify the calls... to pick only good channels". 
        # `generate_fnirs_block_average` in main_pipeline likely doesn't verify `good_channels` argument unless we check/modify it.
        # If we cannot modify `main_pipeline` here, we might skip the old plots if they look bad, 
        # OR we rely on pre-processing bad channel marking.
        
        # Mark bad channels in MNE info for both Raw and Epochs objects
        # This ensures downstream visualization functions respect the bad channel list
        if bad_fnirs_channels:
            mark_bad_channels_in_info(fnirs_epochs, bad_fnirs_channels)
            if processed_fnirs:
                mark_bad_channels_in_info(processed_fnirs, bad_fnirs_channels)

        logger.info("Generating fNIRS block average...")
        block_avg_path = generate_fnirs_block_average_mov_nomov(
            fnirs_epochs, output_path, config, good_channels=good_fnirs_channels
        )
        if block_avg_path:
            viz_paths['fnirs_block_average'] = block_avg_path
        
        # Contrast Map (using anterior ROIs only for motor cortex analysis)
        logger.info("Generating fNIRS contrast map (anterior ROIs)...")
        contrast_path = generate_fnirs_contrast_map_anterior(fnirs_epochs, output_path, config)
        if contrast_path:
            viz_paths['fnirs_contrast'] = contrast_path
    elif modality_mode == ModalityMode.EEG_ONLY:
        logger.info("Skipping fNIRS visualizations (--eeg-only mode)")


    # 7. Quality Assessment (Reuse main pipeline)
    # Note: This will count all channels in raw_fnirs (including misc if any)
    # We can patch n_total_channels after
    # Pass None for modalities that were skipped
    qa_eeg = raw_eeg if modality_mode != ModalityMode.FNIRS_ONLY else None
    qa_fnirs = raw_fnirs if modality_mode != ModalityMode.EEG_ONLY else None
    qa_results = main_pipeline.run_quality_assessment(qa_eeg, qa_fnirs, config)
    
    # FIX TOTAL CHANNELS COUNT (Requirement 3.1, 3.2) - only if fNIRS was processed
    fnirs_report = qa_results.get('fnirs_quality_report') if qa_results else None
    if fnirs_report and raw_fnirs and modality_mode != ModalityMode.EEG_ONLY:
        # Use count_fnirs_channels() to count only fnirs_cw_amplitude channels
        n_fnirs = count_fnirs_channels(raw_fnirs)
        logger.info(f"Correcting total channel count in QA report: {fnirs_report.n_total_channels} -> {n_fnirs}")
        fnirs_report.n_total_channels = n_fnirs

    # 8. Save Full Report
    logger.info("Saving Full Report...")
    main_pipeline.save_full_report(
        qa_results=qa_results,
        eeg_results=eeg_results,
        fnirs_results=fnirs_results,
        multimodal_results=None,
        visualization_paths=viz_paths,
        config=config,
        output_path=output_path
    )
    
    logger.info(f"Sub-012 Analysis Complete. Results available in {output_path}")

if __name__ == "__main__":
    main()
