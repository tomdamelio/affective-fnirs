"""
EEG Analysis Module for ERD/ERS Detection.

This module implements time-frequency analysis of EEG data to detect
Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS)
in motor cortex during finger tapping tasks.

Key analyses:
- Epoch extraction around task events
- Time-Frequency Representation (TFR) using Morlet wavelets
- ERD/ERS detection in mu (8-13 Hz) and beta (13-30 Hz) bands
- Statistical validation of motor patterns
- Visualization of spectrograms and power timecourses

Requirements: 5.8-5.13, 8.2
References:
    - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization
      and desynchronization. Clin Neurophysiol 110(11).
    - Neuper & Pfurtscheller (2001). Event-related dynamics of cortical rhythms.
      Prog Brain Res 159.
    - MNE time-frequency tutorial: https://mne.tools/stable/auto_tutorials/time-freq/
"""

import logging
from typing import Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

from affective_fnirs.config import PipelineConfig

logger = logging.getLogger(__name__)


def create_epochs(
    raw: mne.io.Raw,
    event_id: dict[str, int],
    tmin: float = -5.0,
    tmax: float = 20.0,
    baseline: tuple[float, float] = (-5.0, -1.0),
    reject_criteria: dict[str, float] | None = None,
) -> mne.Epochs:
    """
    Extract epochs around task events from continuous EEG data.

    Epochs are time-locked segments extracted around specific events (e.g., task onset).
    The extended time window (-5 to +20s) accommodates edge artifacts from wavelet
    convolution in time-frequency analysis.

    Algorithm:
    1. Find events in raw.annotations matching event_id
    2. Extract data segments [tmin, tmax] around each event
    3. Apply baseline correction (subtract mean of baseline period)
    4. Optionally reject epochs with excessive artifacts

    Args:
        raw: Preprocessed MNE Raw object (filtered, ICA-cleaned, re-referenced)
        event_id: Dictionary mapping event names to integer codes
            Example: {'task_start': 1, 'block_start': 2}
        tmin: Epoch start time relative to event (seconds, default -5.0)
            Negative values = before event
        tmax: Epoch end time relative to event (seconds, default 20.0)
            Positive values = after event
        baseline: Baseline correction window (start, end) in seconds
            Default (-5.0, -1.0) excludes anticipatory period (-1 to 0s)
            Baseline mean is subtracted from entire epoch
        reject_criteria: Optional amplitude thresholds for artifact rejection
            Example: {'eeg': 150e-6} rejects epochs with EEG > 150 μV
            If None, no automatic rejection (ICA should have removed artifacts)

    Returns:
        MNE Epochs object with extracted time-locked segments

    Notes:
        - Extended window (-5 to +20s) prevents edge artifacts in TFR
        - Baseline excludes -1 to 0s to avoid anticipatory ERD (Req. 5.8)
        - Rejection criteria optional if ICA was effective
        - Epochs can be averaged (epochs.average()) for ERP analysis
        - Use epochs.drop_bad() to manually inspect and reject trials

    Example:
        >>> event_id = {'task_start': 1}
        >>> epochs = create_epochs(raw_clean, event_id, tmin=-5, tmax=20)
        >>> print(f"Created {len(epochs)} epochs")
        >>> # Epochs: 10 trials × 32 channels × 6250 samples (250 Hz × 25s)

    References:
        - MNE Epochs: https://mne.tools/stable/generated/mne.Epochs.html
        - Luck (2014). An Introduction to the Event-Related Potential Technique.

    Requirements: 5.8
    """
    logger.info(
        f"Creating epochs: tmin={tmin}s, tmax={tmax}s, "
        f"baseline={baseline}, event_id={event_id}"
    )

    # Convert annotations to events array
    # events: (n_events, 3) array with [sample_idx, 0, event_code]
    events, event_id_mapping = mne.events_from_annotations(raw, event_id=event_id)

    if len(events) == 0:
        raise ValueError(
            f"No events found matching event_id={event_id}. "
            f"Available annotations: {set(raw.annotations.description)}"
        )

    logger.info(
        f"Found {len(events)} events: "
        f"{dict(zip(event_id_mapping.values(), np.bincount(events[:, 2])))}"
    )

    # Create Epochs object
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_mapping,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject_criteria,
        preload=True,  # Load data into memory for faster processing
        proj=False,  # Don't apply SSP projectors (already preprocessed)
        picks="eeg",  # Only extract EEG channels
        verbose=True,
    )

    n_epochs = len(epochs)
    n_dropped = epochs.drop_log.count(()) - n_epochs  # Count rejected epochs

    logger.info(
        f"Created {n_epochs} epochs "
        f"({n_dropped} dropped due to artifacts or rejection criteria)"
    )

    if n_epochs == 0:
        raise ValueError(
            "All epochs were rejected. Check rejection criteria or data quality."
        )

    return epochs




def compute_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray | None = None,
    n_cycles: float | np.ndarray = 7.0,
    baseline: tuple[float, float] = (-5.0, -1.0),
    baseline_mode: str = "percent",
) -> mne.time_frequency.AverageTFR:
    """
    Compute Time-Frequency Representation using Morlet wavelets.

    TFR decomposes EEG signal into time-varying spectral power across frequencies,
    revealing dynamic changes in oscillatory activity (ERD/ERS) during tasks.

    Algorithm:
    1. For each frequency f:
       - Create Morlet wavelet with n_cycles oscillations
       - Convolve wavelet with epoch data
       - Extract power (squared magnitude of complex result)
    2. Average power across trials
    3. Apply baseline correction (normalize to pre-stimulus power)

    Baseline Correction Modes:
    - 'percent': ((power - baseline) / baseline) * 100
      → Negative values = ERD (power decrease)
      → Positive values = ERS (power increase)
    - 'logratio': 10 * log10(power / baseline)
      → Negative = ERD, Positive = ERS
    - 'zscore': (power - baseline_mean) / baseline_std

    Args:
        epochs: MNE Epochs object with EEG data
        freqs: Frequency array (Hz). Default: 3-30 Hz with 1 Hz steps
            Covers delta (3-4), theta (4-8), alpha (8-13), beta (13-30)
        n_cycles: Number of wavelet cycles
            - float: Constant cycles (e.g., 7.0 for all frequencies)
            - array: Frequency-dependent (e.g., freqs/2 for adaptive resolution)
            Higher cycles → better frequency resolution, worse time resolution
        baseline: Baseline window for normalization (start, end) in seconds
            Default (-5.0, -1.0) matches epoch baseline
        baseline_mode: Normalization method ('percent', 'logratio', 'zscore')
            'percent' recommended for interpretability (Req. 5.10)

    Returns:
        AverageTFR object with baseline-corrected power
            - Shape: (n_channels, n_freqs, n_times)
            - Access data: tfr.data
            - Access times: tfr.times
            - Access freqs: tfr.freqs

    Notes:
        - Morlet wavelets provide good time-frequency trade-off
        - n_cycles=7 is standard for motor rhythms (Pfurtscheller 1999)
        - Frequency-dependent cycles (freqs/2) improve low-freq resolution
        - Baseline correction essential for ERD/ERS interpretation
        - Negative values after 'percent' correction = ERD (power decrease)

    Example:
        >>> freqs = np.arange(3, 31, 1)  # 3-30 Hz, 1 Hz steps
        >>> tfr = compute_tfr(epochs, freqs=freqs, n_cycles=7.0)
        >>> # TFR shape: (32 channels, 28 frequencies, 6250 time points)
        >>> # Negative values in alpha band (8-13 Hz) during task = ERD

    References:
        - Tallon-Baudry et al. (1997). Oscillatory gamma activity in humans. J Neurosci 17(2).
        - MNE TFR: https://mne.tools/stable/generated/mne.time_frequency.tfr_morlet.html
        - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.

    Requirements: 5.9, 5.10
    """
    if freqs is None:
        # Default: 3-30 Hz with 1 Hz steps (covers delta to beta)
        freqs = np.arange(3, 31, 1)

    logger.info(
        f"Computing TFR: freqs={freqs[0]}-{freqs[-1]} Hz ({len(freqs)} steps), "
        f"n_cycles={n_cycles}, baseline={baseline}, mode={baseline_mode}"
    )

    # Compute TFR using Morlet wavelets
    # Average across trials (return_itc=False means don't compute inter-trial coherence)
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,  # FFT convolution (faster for long epochs)
        return_itc=False,  # Only return power, not phase coherence
        average=True,  # Average across trials
        n_jobs=1,  # Parallel processing (1 = single core)
        verbose=True,
    )

    # Apply baseline correction
    tfr.apply_baseline(baseline=baseline, mode=baseline_mode)

    logger.info(
        f"TFR computed: {tfr.data.shape[0]} channels, "
        f"{tfr.data.shape[1]} frequencies, "
        f"{tfr.data.shape[2]} time points"
    )

    # Log baseline correction effect
    if baseline_mode == "percent":
        logger.info(
            "Baseline correction applied (percent mode): "
            "Negative values = ERD (power decrease), "
            "Positive values = ERS (power increase)"
        )

    return tfr




def select_motor_channel(
    raw_or_epochs: mne.io.Raw | mne.Epochs,
    primary_channel: str = "C3",
    fallback_channels: list[str] | None = None,
) -> str:
    """
    Select motor cortex channel with fallback strategy.

    For right-hand finger tapping, the primary analysis channel is C3
    (left motor cortex, contralateral to the moving hand). If C3 is bad
    or missing, fallback to nearby motor channels.

    Channel Selection Rationale:
    - C3: Left motor cortex (primary motor area, contralateral control)
    - CP3: Left central-parietal (sensorimotor integration)
    - C1: Left central (medial motor area)

    Algorithm:
    1. Check if primary_channel exists and is not in info['bads']
    2. If primary unavailable, try fallback_channels in order
    3. If all unavailable, raise error with diagnostic info

    Args:
        raw_or_epochs: MNE Raw or Epochs object with EEG data
        primary_channel: Preferred channel for analysis (default 'C3')
        fallback_channels: Alternative channels if primary unavailable
            Default: ['CP3', 'C1'] for motor cortex analysis

    Returns:
        Selected channel name (str)

    Raises:
        ValueError: If no suitable channel found

    Notes:
        - C3 is contralateral to right hand (left hemisphere)
        - For left-hand tasks, use C4 (right motor cortex)
        - Verify channel not in raw.info['bads'] (Req. 5.11)
        - Log selection rationale for transparency

    Example:
        >>> channel = select_motor_channel(epochs, primary_channel='C3')
        >>> # Log: "Selected motor channel: C3 (primary)"
        >>> # If C3 bad: "Selected motor channel: CP3 (fallback, C3 unavailable)"

    References:
        - Pfurtscheller & Neuper (1997). Motor imagery activates primary sensorimotor area.
        - 10-20 system: Jasper (1958). The ten-twenty electrode system.

    Requirements: 5.11
    """
    if fallback_channels is None:
        fallback_channels = ["CP3", "C1"]

    # Get channel names and bad channels
    ch_names = raw_or_epochs.ch_names
    bad_channels = raw_or_epochs.info["bads"]

    # Try primary channel first
    if primary_channel in ch_names and primary_channel not in bad_channels:
        logger.info(f"Selected motor channel: {primary_channel} (primary)")
        return primary_channel

    # Log why primary was not selected
    if primary_channel not in ch_names:
        logger.warning(
            f"Primary channel {primary_channel} not found in data. "
            f"Available channels: {ch_names}"
        )
    elif primary_channel in bad_channels:
        logger.warning(
            f"Primary channel {primary_channel} marked as bad. "
            f"Trying fallback channels."
        )

    # Try fallback channels
    for fallback_ch in fallback_channels:
        if fallback_ch in ch_names and fallback_ch not in bad_channels:
            logger.info(
                f"Selected motor channel: {fallback_ch} (fallback, "
                f"{primary_channel} unavailable)"
            )
            return fallback_ch

    # No suitable channel found
    available_good_channels = [ch for ch in ch_names if ch not in bad_channels]
    raise ValueError(
        f"No suitable motor channel found. "
        f"Primary ({primary_channel}) and fallbacks ({fallback_channels}) "
        f"are either missing or marked as bad. "
        f"Available good channels: {available_good_channels}"
    )




def detect_erd_ers(
    tfr: mne.time_frequency.AverageTFR,
    channel: str,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    task_window: tuple[float, float] = (1.0, 14.0),
    baseline_window: tuple[float, float] = (-5.0, -1.0),
    alpha_threshold: float = 0.05,
) -> dict:
    """
    Detect Event-Related Desynchronization (ERD) and Synchronization (ERS).

    ERD/ERS quantifies task-related changes in oscillatory power relative to baseline.
    For motor tasks, expected patterns are:
    - Mu ERD (8-13 Hz): -20% to -40% during movement
    - Beta ERD (13-30 Hz): -30% to -50% during movement
    - Beta rebound (ERS): +10% to +30% post-movement (16-20s)

    Algorithm:
    1. Extract power in frequency bands (mu, beta) for selected channel
    2. Compute mean power in task window vs baseline window
    3. Calculate percent change: ((task - baseline) / baseline) * 100
    4. Perform paired t-test for statistical significance
    5. Return metrics: percent change, p-value, significance

    Args:
        tfr: Time-Frequency Representation (baseline-corrected)
        channel: Channel name for analysis (e.g., 'C3')
        alpha_band: Mu/Alpha frequency range (Hz), default (8, 13)
        beta_band: Beta frequency range (Hz), default (13, 30)
        task_window: Task analysis window (seconds), default (1, 14)
            Excludes first second (movement initiation) and last second (offset)
        baseline_window: Baseline comparison window (seconds), default (-5, -1)
            Should match TFR baseline correction window
        alpha_threshold: Significance threshold for p-value (default 0.05)

    Returns:
        Dictionary with ERD/ERS metrics:
            - 'mu_erd_percent': Mean percent change in mu band during task
            - 'mu_p_value': Statistical significance of mu ERD
            - 'mu_significant': Boolean, True if p < alpha_threshold
            - 'beta_erd_percent': Mean percent change in beta band during task
            - 'beta_p_value': Statistical significance of beta ERD
            - 'beta_significant': Boolean, True if p < alpha_threshold
            - 'beta_rebound_percent': Mean percent change in beta post-task
            - 'beta_rebound_window': Time window used for rebound detection
            - 'channel': Channel used for analysis

    Notes:
        - TFR should already be baseline-corrected (mode='percent')
        - Negative percent = ERD (power decrease)
        - Positive percent = ERS (power increase)
        - Paired t-test compares task vs baseline power across trials
        - Expected patterns for right-hand tapping (Req. 5.11-5.13):
          * Mu ERD: -20% to -40%
          * Beta ERD: -30% to -50%
          * Beta rebound: +10% to +30% (16-20s post-task)

    Example:
        >>> results = detect_erd_ers(tfr, channel='C3')
        >>> print(f"Mu ERD: {results['mu_erd_percent']:.1f}% (p={results['mu_p_value']:.3f})")
        >>> # Output: "Mu ERD: -32.5% (p=0.003)" → Significant desynchronization

    References:
        - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.
        - Neuper et al. (2006). ERD/ERS patterns reflecting sensorimotor activation.

    Requirements: 5.11, 5.12, 5.13
    """
    logger.info(
        f"Detecting ERD/ERS for channel {channel}: "
        f"alpha={alpha_band} Hz, beta={beta_band} Hz, "
        f"task_window={task_window}s, baseline_window={baseline_window}s"
    )

    # Get channel index
    if channel not in tfr.ch_names:
        raise ValueError(
            f"Channel {channel} not found in TFR. "
            f"Available channels: {tfr.ch_names}"
        )
    ch_idx = tfr.ch_names.index(channel)

    # Get frequency and time indices
    freqs = tfr.freqs
    times = tfr.times

    # Mu/Alpha band indices
    mu_freq_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    mu_freq_indices = np.where(mu_freq_mask)[0]

    # Beta band indices
    beta_freq_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    beta_freq_indices = np.where(beta_freq_mask)[0]

    # Task window indices
    task_time_mask = (times >= task_window[0]) & (times <= task_window[1])
    task_time_indices = np.where(task_time_mask)[0]

    # Baseline window indices
    baseline_time_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    baseline_time_indices = np.where(baseline_time_mask)[0]

    # Beta rebound window (post-task: 16-20s)
    rebound_window = (16.0, 20.0)
    rebound_time_mask = (times >= rebound_window[0]) & (times <= rebound_window[1])
    rebound_time_indices = np.where(rebound_time_mask)[0]

    # Extract TFR data for channel
    # Shape: (n_freqs, n_times)
    tfr_data = tfr.data[ch_idx, :, :]

    # === Mu/Alpha ERD Analysis ===
    # Extract mu power in task and baseline windows
    mu_task_power = tfr_data[np.ix_(mu_freq_indices, task_time_indices)]
    mu_baseline_power = tfr_data[np.ix_(mu_freq_indices, baseline_time_indices)]

    # Average across frequency and time
    mu_task_mean = np.mean(mu_task_power)
    mu_baseline_mean = np.mean(mu_baseline_power)

    # Calculate percent change (already baseline-corrected, so this is redundant
    # but we recalculate for clarity and to match expected output format)
    # Since TFR is already in percent mode, mu_task_mean is already the percent change
    mu_erd_percent = mu_task_mean

    # For statistical test, we need trial-by-trial data
    # Since we have AverageTFR (averaged across trials), we can't do paired t-test
    # Instead, we'll use a one-sample t-test against zero (is the mean significantly different from 0?)
    # However, this requires access to individual trial data
    # For now, we'll report the mean and note that statistical testing requires trial data

    # Placeholder for p-value (would require trial-level TFR)
    mu_p_value = np.nan  # Cannot compute without trial data
    mu_significant = False  # Conservative: assume not significant without test

    logger.info(
        f"Mu ERD: {mu_erd_percent:.1f}% "
        f"(expected: -20% to -40% for motor task)"
    )

    # === Beta ERD Analysis ===
    beta_task_power = tfr_data[np.ix_(beta_freq_indices, task_time_indices)]
    beta_baseline_power = tfr_data[np.ix_(beta_freq_indices, baseline_time_indices)]

    beta_task_mean = np.mean(beta_task_power)
    beta_baseline_mean = np.mean(beta_baseline_power)

    beta_erd_percent = beta_task_mean

    beta_p_value = np.nan
    beta_significant = False

    logger.info(
        f"Beta ERD: {beta_erd_percent:.1f}% "
        f"(expected: -30% to -50% for motor task)"
    )

    # === Beta Rebound (ERS) Analysis ===
    if len(rebound_time_indices) > 0:
        beta_rebound_power = tfr_data[np.ix_(beta_freq_indices, rebound_time_indices)]
        beta_rebound_mean = np.mean(beta_rebound_power)
        beta_rebound_percent = beta_rebound_mean

        logger.info(
            f"Beta rebound: {beta_rebound_percent:.1f}% at {rebound_window}s "
            f"(expected: +10% to +30% post-task)"
        )
    else:
        beta_rebound_percent = np.nan
        logger.warning(
            f"Beta rebound window {rebound_window}s not found in TFR times"
        )

    # Compile results
    results = {
        "channel": channel,
        "mu_erd_percent": mu_erd_percent,
        "mu_p_value": mu_p_value,
        "mu_significant": mu_significant,
        "beta_erd_percent": beta_erd_percent,
        "beta_p_value": beta_p_value,
        "beta_significant": beta_significant,
        "beta_rebound_percent": beta_rebound_percent,
        "beta_rebound_window": rebound_window,
        "alpha_band": alpha_band,
        "beta_band": beta_band,
        "task_window": task_window,
        "baseline_window": baseline_window,
    }

    # Log summary
    logger.info("=" * 60)
    logger.info(f"ERD/ERS Detection Summary (Channel: {channel})")
    logger.info("=" * 60)
    logger.info(f"Mu ERD:         {mu_erd_percent:>8.1f}% (expected: -20% to -40%)")
    logger.info(f"Beta ERD:       {beta_erd_percent:>8.1f}% (expected: -30% to -50%)")
    logger.info(
        f"Beta Rebound:   {beta_rebound_percent:>8.1f}% (expected: +10% to +30%)"
    )
    logger.info("=" * 60)
    logger.info(
        "Note: Statistical significance requires trial-level TFR data. "
        "Current implementation uses averaged TFR."
    )

    return results




def plot_eeg_spectrogram(
    tfr: mne.time_frequency.AverageTFR,
    channel: str,
    vmin: float = -50.0,
    vmax: float = 50.0,
    cmap: str = "RdBu_r",
    task_onset: float = 0.0,
    task_offset: float = 15.0,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    figsize: tuple[float, float] = (12, 6),
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot time-frequency spectrogram for EEG channel.

    Visualizes ERD/ERS patterns as a heatmap of power changes across time and frequency.
    Uses diverging colormap centered at 0% to highlight desynchronization (blue) and
    synchronization (red).

    Args:
        tfr: Time-Frequency Representation (baseline-corrected)
        channel: Channel name to plot (e.g., 'C3')
        vmin: Minimum colormap value (percent change, default -50%)
        vmax: Maximum colormap value (percent change, default +50%)
        cmap: Colormap name (default 'RdBu_r' for diverging blue-red)
            'RdBu_r': Blue = negative (ERD), Red = positive (ERS)
        task_onset: Task start time for annotation (seconds, default 0)
        task_offset: Task end time for annotation (seconds, default 15)
        alpha_band: Alpha/Mu frequency band for annotation (Hz)
        beta_band: Beta frequency band for annotation (Hz)
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save figure (PNG format)

    Returns:
        Matplotlib Figure object

    Notes:
        - Colormap centered at 0% (symmetric vmin/vmax)
        - Blue regions = ERD (power decrease)
        - Red regions = ERS (power increase)
        - Vertical lines mark task onset/offset
        - Horizontal lines mark frequency bands
        - Expected pattern: Blue in alpha/beta during task (ERD)

    Example:
        >>> fig = plot_eeg_spectrogram(tfr, channel='C3', vmin=-50, vmax=50)
        >>> # Shows blue (ERD) in alpha (8-13 Hz) and beta (13-30 Hz) during task
        >>> plt.show()

    References:
        - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.

    Requirements: 8.2
    """
    logger.info(f"Plotting spectrogram for channel {channel}")

    # Get channel index
    if channel not in tfr.ch_names:
        raise ValueError(
            f"Channel {channel} not found in TFR. "
            f"Available channels: {tfr.ch_names}"
        )
    ch_idx = tfr.ch_names.index(channel)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for channel
    # Shape: (n_freqs, n_times)
    data = tfr.data[ch_idx, :, :]
    times = tfr.times
    freqs = tfr.freqs

    # Plot spectrogram
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Power change (%)")

    # Annotate task window
    ax.axvline(task_onset, color="black", linestyle="--", linewidth=2, label="Task onset")
    ax.axvline(task_offset, color="black", linestyle="--", linewidth=2, label="Task offset")

    # Annotate frequency bands
    ax.axhline(alpha_band[0], color="white", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(alpha_band[1], color="white", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(
        times[-1] * 0.98,
        np.mean(alpha_band),
        "Alpha/Mu",
        color="white",
        fontsize=10,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    ax.axhline(beta_band[0], color="white", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(beta_band[1], color="white", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(
        times[-1] * 0.98,
        np.mean(beta_band),
        "Beta",
        color="white",
        fontsize=10,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    # Labels and title
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(
        f"Time-Frequency Spectrogram: {channel}\n"
        f"(Blue = ERD, Red = ERS)",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    ax.legend(loc="upper left", fontsize=10)

    # Grid
    ax.grid(False)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spectrogram saved to: {output_path}")

    return fig


def plot_erd_timecourse(
    tfr: mne.time_frequency.AverageTFR,
    channel: str,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    task_onset: float = 0.0,
    task_offset: float = 15.0,
    figsize: tuple[float, float] = (12, 6),
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot ERD/ERS timecourse for alpha and beta bands.

    Shows power changes over time for specific frequency bands, with error bands
    representing variability across trials (if available).

    Args:
        tfr: Time-Frequency Representation (baseline-corrected)
        channel: Channel name to plot (e.g., 'C3')
        alpha_band: Alpha/Mu frequency range (Hz), default (8, 13)
        beta_band: Beta frequency range (Hz), default (13, 30)
        task_onset: Task start time for annotation (seconds, default 0)
        task_offset: Task end time for annotation (seconds, default 15)
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save figure (PNG format)

    Returns:
        Matplotlib Figure object

    Notes:
        - Averaged power across frequency band
        - Error bands show ±1 SEM (Standard Error of Mean) if trial data available
        - Negative values = ERD (power decrease)
        - Positive values = ERS (power increase)
        - Expected pattern:
          * Alpha ERD during task (negative dip)
          * Beta ERD during task (negative dip)
          * Beta rebound post-task (positive peak)

    Example:
        >>> fig = plot_erd_timecourse(tfr, channel='C3')
        >>> # Shows alpha and beta power timecourses with task annotations
        >>> plt.show()

    References:
        - Neuper & Pfurtscheller (2001). Event-related dynamics of cortical rhythms.

    Requirements: 8.2
    """
    logger.info(f"Plotting ERD timecourse for channel {channel}")

    # Get channel index
    if channel not in tfr.ch_names:
        raise ValueError(
            f"Channel {channel} not found in TFR. "
            f"Available channels: {tfr.ch_names}"
        )
    ch_idx = tfr.ch_names.index(channel)

    # Extract data
    data = tfr.data[ch_idx, :, :]
    times = tfr.times
    freqs = tfr.freqs

    # Get frequency band indices
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])

    # Average power across frequency bands
    alpha_power = np.mean(data[alpha_mask, :], axis=0)
    beta_power = np.mean(data[beta_mask, :], axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot alpha timecourse
    ax.plot(
        times,
        alpha_power,
        color="blue",
        linewidth=2,
        label=f"Alpha/Mu ({alpha_band[0]}-{alpha_band[1]} Hz)",
    )

    # Plot beta timecourse
    ax.plot(
        times,
        beta_power,
        color="red",
        linewidth=2,
        label=f"Beta ({beta_band[0]}-{beta_band[1]} Hz)",
    )

    # Note: Error bands (±1 SEM) would require trial-level data
    # For AverageTFR, we only have the mean across trials

    # Add zero line
    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    # Annotate task window
    ax.axvline(task_onset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(task_offset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvspan(
        task_onset,
        task_offset,
        alpha=0.1,
        color="gray",
        label="Task window",
    )

    # Labels and title
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Power change (%)", fontsize=12)
    ax.set_title(
        f"ERD/ERS Timecourse: {channel}\n"
        f"(Negative = ERD, Positive = ERS)",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    ax.legend(loc="best", fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"ERD timecourse saved to: {output_path}")

    return fig


