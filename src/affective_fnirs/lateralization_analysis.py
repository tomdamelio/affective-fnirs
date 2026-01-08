"""
Lateralization Analysis for Motor Tasks.

This module implements contralateral ERD analysis for finger tapping tasks,
comparing LEFT vs RIGHT hand conditions to verify expected lateralization patterns.

Expected patterns:
- LEFT hand movement → ERD in C4 (right hemisphere, contralateral)
- RIGHT hand movement → ERD in C3 (left hemisphere, contralateral)
- NOTHING condition → No significant ERD (baseline control)

References:
    - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.
    - Neuper & Pfurtscheller (2001). Motor imagery and direct brain-computer communication.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class LateralizationResult:
    """Results from lateralization analysis."""

    # Condition-specific ERD values
    left_c3_alpha_erd: float  # LEFT condition, C3 (ipsilateral)
    left_c4_alpha_erd: float  # LEFT condition, C4 (contralateral)
    right_c3_alpha_erd: float  # RIGHT condition, C3 (contralateral)
    right_c4_alpha_erd: float  # RIGHT condition, C4 (ipsilateral)
    nothing_c3_alpha_erd: float  # NOTHING condition, C3
    nothing_c4_alpha_erd: float  # NOTHING condition, C4

    # Same for beta band
    left_c3_beta_erd: float
    left_c4_beta_erd: float
    right_c3_beta_erd: float
    right_c4_beta_erd: float
    nothing_c3_beta_erd: float
    nothing_c4_beta_erd: float

    # Lateralization indices
    left_lateralization_alpha: float  # (C4 - C3) for LEFT condition
    right_lateralization_alpha: float  # (C3 - C4) for RIGHT condition
    left_lateralization_beta: float
    right_lateralization_beta: float

    # Statistical comparisons
    left_vs_nothing_c4_p: float  # LEFT C4 vs NOTHING C4
    right_vs_nothing_c3_p: float  # RIGHT C3 vs NOTHING C3
    left_contralateral_vs_ipsilateral_p: float  # LEFT: C4 vs C3
    right_contralateral_vs_ipsilateral_p: float  # RIGHT: C3 vs C4

    # Validation flags
    left_shows_contralateral_erd: bool
    right_shows_contralateral_erd: bool
    lateralization_pattern_valid: bool

    # Number of trials per condition
    n_left_trials: int
    n_right_trials: int
    n_nothing_trials: int


def compute_lateralization_analysis(
    raw: mne.io.Raw,
    alpha_band: Tuple[float, float] = (8.0, 13.0),
    beta_band: Tuple[float, float] = (13.0, 30.0),
    tmin: float = -2.0,
    tmax: float = 15.0,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    task_window: Tuple[float, float] = (0.5, 5.0),
    freqs: np.ndarray | None = None,
) -> LateralizationResult:
    """
    Compute lateralization analysis comparing LEFT, RIGHT, and NOTHING conditions.

    This analysis verifies the expected contralateral ERD pattern:
    - LEFT hand → stronger ERD in C4 (right hemisphere)
    - RIGHT hand → stronger ERD in C3 (left hemisphere)

    Args:
        raw: Preprocessed EEG Raw object with annotations
        alpha_band: Alpha/Mu frequency range (Hz)
        beta_band: Beta frequency range (Hz)
        tmin: Epoch start time (seconds)
        tmax: Epoch end time (seconds)
        baseline: Baseline window for correction (seconds)
        task_window: Window for ERD calculation (seconds post-stimulus)
        freqs: Frequencies for TFR (default: 4-35 Hz)

    Returns:
        LateralizationResult with all metrics and validation flags
    """
    logger.info("=" * 70)
    logger.info("LATERALIZATION ANALYSIS")
    logger.info("=" * 70)

    if freqs is None:
        freqs = np.arange(4, 36, 1)

    # Get events
    events, event_id = mne.events_from_annotations(raw)

    # Identify condition event IDs
    left_ids = {k: v for k, v in event_id.items() if "LEFT" in k.upper()}
    right_ids = {k: v for k, v in event_id.items() if "RIGHT" in k.upper()}
    nothing_ids = {k: v for k, v in event_id.items() if "NOTHING" in k.upper()}

    logger.info(f"Event IDs - LEFT: {left_ids}, RIGHT: {right_ids}, NOTHING: {nothing_ids}")

    # Verify we have motor channels
    motor_channels = ["C3", "C4"]
    for ch in motor_channels:
        if ch not in raw.ch_names:
            raise ValueError(f"Motor channel {ch} not found in data")

    # Create epochs for each condition
    def create_condition_epochs(event_ids: dict, condition_name: str) -> mne.Epochs | None:
        if not event_ids:
            logger.warning(f"No events found for {condition_name}")
            return None

        try:
            # Note: No baseline correction on epochs - we apply it on TFR instead
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_ids,
                tmin=tmin,
                tmax=tmax,
                baseline=None,  # No baseline on epochs, apply on TFR
                picks=["C3", "C4"],
                preload=True,
                reject=None,
                verbose=False,
            )
            logger.info(f"{condition_name}: {len(epochs)} epochs")
            return epochs
        except Exception as e:
            logger.warning(f"Failed to create epochs for {condition_name}: {e}")
            return None

    epochs_left = create_condition_epochs(left_ids, "LEFT")
    epochs_right = create_condition_epochs(right_ids, "RIGHT")
    epochs_nothing = create_condition_epochs(nothing_ids, "NOTHING")

    # Compute TFR for each condition
    def compute_condition_tfr(
        epochs: mne.Epochs | None, condition_name: str
    ) -> mne.time_frequency.AverageTFR | None:
        if epochs is None or len(epochs) == 0:
            return None

        n_cycles = freqs / 2.0  # Adaptive cycles

        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False,
        )
        tfr.apply_baseline(baseline=baseline, mode="percent")
        logger.info(f"TFR computed for {condition_name}")
        return tfr

    tfr_left = compute_condition_tfr(epochs_left, "LEFT")
    tfr_right = compute_condition_tfr(epochs_right, "RIGHT")
    tfr_nothing = compute_condition_tfr(epochs_nothing, "NOTHING")

    # Extract ERD values for each condition and channel
    def extract_erd(
        tfr: mne.time_frequency.AverageTFR | None,
        channel: str,
        freq_band: Tuple[float, float],
    ) -> float:
        if tfr is None:
            return np.nan

        ch_idx = tfr.ch_names.index(channel)
        freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
        time_mask = (tfr.times >= task_window[0]) & (tfr.times <= task_window[1])

        data = tfr.data[ch_idx, :, :]
        # MNE's mode='percent' returns ratio (e.g., -0.5 for -50%)
        # Multiply by 100 to get percentage
        erd = np.mean(data[freq_mask, :][:, time_mask]) * 100
        return erd

    # Extract all ERD values
    left_c3_alpha = extract_erd(tfr_left, "C3", alpha_band)
    left_c4_alpha = extract_erd(tfr_left, "C4", alpha_band)
    right_c3_alpha = extract_erd(tfr_right, "C3", alpha_band)
    right_c4_alpha = extract_erd(tfr_right, "C4", alpha_band)
    nothing_c3_alpha = extract_erd(tfr_nothing, "C3", alpha_band)
    nothing_c4_alpha = extract_erd(tfr_nothing, "C4", alpha_band)

    left_c3_beta = extract_erd(tfr_left, "C3", beta_band)
    left_c4_beta = extract_erd(tfr_left, "C4", beta_band)
    right_c3_beta = extract_erd(tfr_right, "C3", beta_band)
    right_c4_beta = extract_erd(tfr_right, "C4", beta_band)
    nothing_c3_beta = extract_erd(tfr_nothing, "C3", beta_band)
    nothing_c4_beta = extract_erd(tfr_nothing, "C4", beta_band)

    # Compute lateralization indices
    # For LEFT: contralateral (C4) should show MORE ERD (more negative) than ipsilateral (C3)
    # Lateralization = C4 - C3 (should be negative for LEFT)
    left_lat_alpha = left_c4_alpha - left_c3_alpha
    right_lat_alpha = right_c3_alpha - right_c4_alpha  # C3 - C4 for RIGHT
    left_lat_beta = left_c4_beta - left_c3_beta
    right_lat_beta = right_c3_beta - right_c4_beta

    # Statistical tests (using trial-level data if available)
    def compute_trial_erd(
        epochs: mne.Epochs | None,
        channel: str,
        freq_band: Tuple[float, float],
    ) -> np.ndarray:
        """Compute ERD for each trial (in percentage)."""
        if epochs is None or len(epochs) == 0:
            return np.array([])

        n_cycles = freqs / 2.0
        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=False,  # Keep trial dimension
            verbose=False,
        )
        tfr.apply_baseline(baseline=baseline, mode="percent")

        ch_idx = tfr.ch_names.index(channel)
        freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
        time_mask = (tfr.times >= task_window[0]) & (tfr.times <= task_window[1])

        # Shape: (n_trials, n_channels, n_freqs, n_times)
        data = tfr.data[:, ch_idx, :, :]
        # Average over freq and time for each trial
        # Multiply by 100 to convert ratio to percentage
        trial_erds = np.mean(data[:, freq_mask, :][:, :, time_mask], axis=(1, 2)) * 100
        return trial_erds

    # Compute trial-level ERD for statistical tests
    left_c4_trials = compute_trial_erd(epochs_left, "C4", alpha_band)
    left_c3_trials = compute_trial_erd(epochs_left, "C3", alpha_band)
    right_c3_trials = compute_trial_erd(epochs_right, "C3", alpha_band)
    right_c4_trials = compute_trial_erd(epochs_right, "C4", alpha_band)
    nothing_c3_trials = compute_trial_erd(epochs_nothing, "C3", alpha_band)
    nothing_c4_trials = compute_trial_erd(epochs_nothing, "C4", alpha_band)

    # Statistical comparisons
    def safe_ttest(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return np.nan
        try:
            _, p = stats.ttest_ind(a, b)
            return p
        except Exception:
            return np.nan

    def safe_paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(a) != len(b):
            return np.nan
        try:
            _, p = stats.ttest_rel(a, b)
            return p
        except Exception:
            return np.nan

    # LEFT C4 vs NOTHING C4 (should be significant if LEFT shows ERD)
    left_vs_nothing_c4_p = safe_ttest(left_c4_trials, nothing_c4_trials)

    # RIGHT C3 vs NOTHING C3 (should be significant if RIGHT shows ERD)
    right_vs_nothing_c3_p = safe_ttest(right_c3_trials, nothing_c3_trials)

    # LEFT: C4 vs C3 (contralateral vs ipsilateral)
    left_contra_vs_ipsi_p = safe_paired_ttest(left_c4_trials, left_c3_trials)

    # RIGHT: C3 vs C4 (contralateral vs ipsilateral)
    right_contra_vs_ipsi_p = safe_paired_ttest(right_c3_trials, right_c4_trials)

    # Validation flags
    # LEFT shows contralateral ERD if C4 is more negative than C3
    left_shows_contra = left_c4_alpha < left_c3_alpha and left_c4_alpha < -5

    # RIGHT shows contralateral ERD if C3 is more negative than C4
    right_shows_contra = right_c3_alpha < right_c4_alpha and right_c3_alpha < -5

    # Overall pattern is valid if both conditions show expected lateralization
    pattern_valid = left_shows_contra and right_shows_contra

    # Log results
    logger.info("\n" + "=" * 70)
    logger.info("LATERALIZATION RESULTS")
    logger.info("=" * 70)

    logger.info("\n--- ALPHA BAND ERD (%) ---")
    logger.info(f"{'Condition':<12} {'C3 (left)':>12} {'C4 (right)':>12} {'Lateralization':>15}")
    logger.info("-" * 55)
    logger.info(f"{'LEFT':<12} {left_c3_alpha:>12.1f} {left_c4_alpha:>12.1f} {left_lat_alpha:>15.1f}")
    logger.info(f"{'RIGHT':<12} {right_c3_alpha:>12.1f} {right_c4_alpha:>12.1f} {right_lat_alpha:>15.1f}")
    logger.info(f"{'NOTHING':<12} {nothing_c3_alpha:>12.1f} {nothing_c4_alpha:>12.1f} {'N/A':>15}")

    logger.info("\n--- BETA BAND ERD (%) ---")
    logger.info(f"{'Condition':<12} {'C3 (left)':>12} {'C4 (right)':>12} {'Lateralization':>15}")
    logger.info("-" * 55)
    logger.info(f"{'LEFT':<12} {left_c3_beta:>12.1f} {left_c4_beta:>12.1f} {left_lat_beta:>15.1f}")
    logger.info(f"{'RIGHT':<12} {right_c3_beta:>12.1f} {right_c4_beta:>12.1f} {right_lat_beta:>15.1f}")
    logger.info(f"{'NOTHING':<12} {nothing_c3_beta:>12.1f} {nothing_c4_beta:>12.1f} {'N/A':>15}")

    logger.info("\n--- STATISTICAL TESTS ---")
    logger.info(f"LEFT C4 vs NOTHING C4: p = {left_vs_nothing_c4_p:.4f}")
    logger.info(f"RIGHT C3 vs NOTHING C3: p = {right_vs_nothing_c3_p:.4f}")
    logger.info(f"LEFT: C4 vs C3 (paired): p = {left_contra_vs_ipsi_p:.4f}")
    logger.info(f"RIGHT: C3 vs C4 (paired): p = {right_contra_vs_ipsi_p:.4f}")

    logger.info("\n--- VALIDATION ---")
    logger.info(f"LEFT shows contralateral ERD (C4): {left_shows_contra}")
    logger.info(f"RIGHT shows contralateral ERD (C3): {right_shows_contra}")
    logger.info(f"Overall lateralization pattern valid: {pattern_valid}")

    if pattern_valid:
        logger.info("\n✅ LATERALIZATION PATTERN VALIDATED")
        logger.info("   Motor task shows expected contralateral ERD pattern")
    else:
        logger.warning("\n⚠️ LATERALIZATION PATTERN NOT VALIDATED")
        if not left_shows_contra:
            logger.warning("   - LEFT condition does not show expected C4 ERD")
        if not right_shows_contra:
            logger.warning("   - RIGHT condition does not show expected C3 ERD")

    return LateralizationResult(
        left_c3_alpha_erd=left_c3_alpha,
        left_c4_alpha_erd=left_c4_alpha,
        right_c3_alpha_erd=right_c3_alpha,
        right_c4_alpha_erd=right_c4_alpha,
        nothing_c3_alpha_erd=nothing_c3_alpha,
        nothing_c4_alpha_erd=nothing_c4_alpha,
        left_c3_beta_erd=left_c3_beta,
        left_c4_beta_erd=left_c4_beta,
        right_c3_beta_erd=right_c3_beta,
        right_c4_beta_erd=right_c4_beta,
        nothing_c3_beta_erd=nothing_c3_beta,
        nothing_c4_beta_erd=nothing_c4_beta,
        left_lateralization_alpha=left_lat_alpha,
        right_lateralization_alpha=right_lat_alpha,
        left_lateralization_beta=left_lat_beta,
        right_lateralization_beta=right_lat_beta,
        left_vs_nothing_c4_p=left_vs_nothing_c4_p,
        right_vs_nothing_c3_p=right_vs_nothing_c3_p,
        left_contralateral_vs_ipsilateral_p=left_contra_vs_ipsi_p,
        right_contralateral_vs_ipsilateral_p=right_contra_vs_ipsi_p,
        left_shows_contralateral_erd=left_shows_contra,
        right_shows_contralateral_erd=right_shows_contra,
        lateralization_pattern_valid=pattern_valid,
        n_left_trials=len(epochs_left) if epochs_left else 0,
        n_right_trials=len(epochs_right) if epochs_right else 0,
        n_nothing_trials=len(epochs_nothing) if epochs_nothing else 0,
    )


@dataclass
class LateralizationTimeCourseData:
    """Time-course data for lateralization analysis."""

    times: np.ndarray
    left_c3_alpha: np.ndarray
    left_c4_alpha: np.ndarray
    right_c3_alpha: np.ndarray
    right_c4_alpha: np.ndarray
    nothing_c3_alpha: np.ndarray
    nothing_c4_alpha: np.ndarray
    left_c3_beta: np.ndarray
    left_c4_beta: np.ndarray
    right_c3_beta: np.ndarray
    right_c4_beta: np.ndarray
    nothing_c3_beta: np.ndarray
    nothing_c4_beta: np.ndarray


def compute_lateralization_timecourse(
    raw: mne.io.Raw,
    alpha_band: Tuple[float, float] = (8.0, 13.0),
    beta_band: Tuple[float, float] = (13.0, 30.0),
    tmin: float = -2.0,
    tmax: float = 15.0,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    freqs: np.ndarray | None = None,
) -> LateralizationTimeCourseData:
    """
    Compute time-course of ERD for lateralization analysis.

    Args:
        raw: Preprocessed EEG Raw object with annotations
        alpha_band: Alpha/Mu frequency range (Hz)
        beta_band: Beta frequency range (Hz)
        tmin: Epoch start time (seconds)
        tmax: Epoch end time (seconds)
        baseline: Baseline window for correction (seconds)
        freqs: Frequencies for TFR (default: 4-35 Hz)

    Returns:
        LateralizationTimeCourseData with time-course arrays for each condition/channel
    """
    if freqs is None:
        freqs = np.arange(4, 36, 1)

    events, event_id = mne.events_from_annotations(raw)

    left_ids = {k: v for k, v in event_id.items() if "LEFT" in k.upper()}
    right_ids = {k: v for k, v in event_id.items() if "RIGHT" in k.upper()}
    nothing_ids = {k: v for k, v in event_id.items() if "NOTHING" in k.upper()}

    def create_epochs(event_ids: dict) -> mne.Epochs | None:
        if not event_ids:
            return None
        return mne.Epochs(
            raw, events, event_id=event_ids,
            tmin=tmin, tmax=tmax, baseline=None,
            picks=["C3", "C4"], preload=True, reject=None, verbose=False,
        )

    def compute_tfr_timecourse(epochs: mne.Epochs | None) -> mne.time_frequency.AverageTFR | None:
        if epochs is None or len(epochs) == 0:
            return None
        n_cycles = freqs / 2.0
        tfr = mne.time_frequency.tfr_morlet(
            epochs, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False,
        )
        tfr.apply_baseline(baseline=baseline, mode="percent")
        return tfr

    def extract_band_timecourse(tfr: mne.time_frequency.AverageTFR | None, channel: str, freq_band: Tuple[float, float]) -> np.ndarray:
        if tfr is None:
            return np.array([])
        ch_idx = tfr.ch_names.index(channel)
        freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
        # Average over frequencies, keep time dimension, multiply by 100 for percentage
        return np.mean(tfr.data[ch_idx, freq_mask, :], axis=0) * 100

    epochs_left = create_epochs(left_ids)
    epochs_right = create_epochs(right_ids)
    epochs_nothing = create_epochs(nothing_ids)

    tfr_left = compute_tfr_timecourse(epochs_left)
    tfr_right = compute_tfr_timecourse(epochs_right)
    tfr_nothing = compute_tfr_timecourse(epochs_nothing)

    # Get times from any available TFR
    times = tfr_left.times if tfr_left is not None else (tfr_right.times if tfr_right is not None else np.array([]))

    return LateralizationTimeCourseData(
        times=times,
        left_c3_alpha=extract_band_timecourse(tfr_left, "C3", alpha_band),
        left_c4_alpha=extract_band_timecourse(tfr_left, "C4", alpha_band),
        right_c3_alpha=extract_band_timecourse(tfr_right, "C3", alpha_band),
        right_c4_alpha=extract_band_timecourse(tfr_right, "C4", alpha_band),
        nothing_c3_alpha=extract_band_timecourse(tfr_nothing, "C3", alpha_band),
        nothing_c4_alpha=extract_band_timecourse(tfr_nothing, "C4", alpha_band),
        left_c3_beta=extract_band_timecourse(tfr_left, "C3", beta_band),
        left_c4_beta=extract_band_timecourse(tfr_left, "C4", beta_band),
        right_c3_beta=extract_band_timecourse(tfr_right, "C3", beta_band),
        right_c4_beta=extract_band_timecourse(tfr_right, "C4", beta_band),
        nothing_c3_beta=extract_band_timecourse(tfr_nothing, "C3", beta_band),
        nothing_c4_beta=extract_band_timecourse(tfr_nothing, "C4", beta_band),
    )


def plot_lateralization_timecourse(
    timecourse: LateralizationTimeCourseData,
    result: LateralizationResult,
) -> plt.Figure:
    """
    Plot time-course lineplots for lateralization analysis.

    Creates a 2x2 figure with:
    1. C3 Alpha: LEFT vs RIGHT vs NOTHING
    2. C4 Alpha: LEFT vs RIGHT vs NOTHING
    3. C3 Beta: LEFT vs RIGHT vs NOTHING
    4. C4 Beta: LEFT vs RIGHT vs NOTHING

    Args:
        timecourse: LateralizationTimeCourseData with time-course arrays
        result: LateralizationResult for annotations

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    times = timecourse.times

    colors = {"LEFT": "#E74C3C", "RIGHT": "#3498DB", "NOTHING": "#95A5A6"}
    linewidth = 2

    # Plot 1: C3 Alpha
    ax = axes[0, 0]
    ax.plot(times, timecourse.left_c3_alpha, color=colors["LEFT"], linewidth=linewidth, label="LEFT")
    ax.plot(times, timecourse.right_c3_alpha, color=colors["RIGHT"], linewidth=linewidth, label="RIGHT")
    ax.plot(times, timecourse.nothing_c3_alpha, color=colors["NOTHING"], linewidth=linewidth, label="NOTHING")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="Stimulus onset")
    ax.fill_between([0.5, 5], -100, 100, alpha=0.1, color="green", label="Task window")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("C3 (Left Hemisphere) - Alpha Band (8-13 Hz)\nExpected: RIGHT hand → C3 ERD")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: C4 Alpha
    ax = axes[0, 1]
    ax.plot(times, timecourse.left_c4_alpha, color=colors["LEFT"], linewidth=linewidth, label="LEFT")
    ax.plot(times, timecourse.right_c4_alpha, color=colors["RIGHT"], linewidth=linewidth, label="RIGHT")
    ax.plot(times, timecourse.nothing_c4_alpha, color=colors["NOTHING"], linewidth=linewidth, label="NOTHING")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.fill_between([0.5, 5], -100, 100, alpha=0.1, color="green")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("C4 (Right Hemisphere) - Alpha Band (8-13 Hz)\nExpected: LEFT hand → C4 ERD")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: C3 Beta
    ax = axes[1, 0]
    ax.plot(times, timecourse.left_c3_beta, color=colors["LEFT"], linewidth=linewidth, label="LEFT")
    ax.plot(times, timecourse.right_c3_beta, color=colors["RIGHT"], linewidth=linewidth, label="RIGHT")
    ax.plot(times, timecourse.nothing_c3_beta, color=colors["NOTHING"], linewidth=linewidth, label="NOTHING")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.fill_between([0.5, 5], -100, 100, alpha=0.1, color="green")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("C3 (Left Hemisphere) - Beta Band (13-30 Hz)\nExpected: RIGHT hand → C3 ERD")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: C4 Beta
    ax = axes[1, 1]
    ax.plot(times, timecourse.left_c4_beta, color=colors["LEFT"], linewidth=linewidth, label="LEFT")
    ax.plot(times, timecourse.right_c4_beta, color=colors["RIGHT"], linewidth=linewidth, label="RIGHT")
    ax.plot(times, timecourse.nothing_c4_beta, color=colors["NOTHING"], linewidth=linewidth, label="NOTHING")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.fill_between([0.5, 5], -100, 100, alpha=0.1, color="green")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("C4 (Right Hemisphere) - Beta Band (13-30 Hz)\nExpected: LEFT hand → C4 ERD")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Lateralization Time-Course Analysis\n"
        f"LEFT valid: {'✓' if result.left_shows_contralateral_erd else '✗'} | "
        f"RIGHT valid: {'✓' if result.right_shows_contralateral_erd else '✗'}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_lateralization_barplot(result: LateralizationResult) -> plt.Figure:
    """
    Plot bar chart comparing ERD across conditions.

    Args:
        result: LateralizationResult from compute_lateralization_analysis

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conditions = ["LEFT", "RIGHT", "NOTHING"]
    x = np.arange(len(conditions))
    width = 0.35

    # Alpha ERD
    ax = axes[0]
    c3_alpha = [result.left_c3_alpha_erd, result.right_c3_alpha_erd, result.nothing_c3_alpha_erd]
    c4_alpha = [result.left_c4_alpha_erd, result.right_c4_alpha_erd, result.nothing_c4_alpha_erd]

    bars1 = ax.bar(x - width/2, c3_alpha, width, label="C3 (left)", color="#3498DB", alpha=0.8)
    bars2 = ax.bar(x + width/2, c4_alpha, width, label="C4 (right)", color="#E74C3C", alpha=0.8)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("Alpha Band (8-13 Hz)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    # Beta ERD
    ax = axes[1]
    c3_beta = [result.left_c3_beta_erd, result.right_c3_beta_erd, result.nothing_c3_beta_erd]
    c4_beta = [result.left_c4_beta_erd, result.right_c4_beta_erd, result.nothing_c4_beta_erd]

    bars1 = ax.bar(x - width/2, c3_beta, width, label="C3 (left)", color="#3498DB", alpha=0.8)
    bars2 = ax.bar(x + width/2, c4_beta, width, label="C4 (right)", color="#E74C3C", alpha=0.8)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("ERD/ERS (%)")
    ax.set_title("Beta Band (13-30 Hz)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    plt.suptitle("Lateralization ERD Comparison by Condition", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_lateralization_results(
    result: LateralizationResult,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot lateralization analysis results (legacy function for backward compatibility).

    Args:
        result: LateralizationResult from compute_lateralization_analysis
        output_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig = plot_lateralization_barplot(result)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Lateralization plot saved to: {output_path}")

    return fig
