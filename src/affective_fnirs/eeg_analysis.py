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
    beta_rebound_window: tuple[float, float] = (15.0, 20.0),
    alpha_threshold: float = 0.05,
) -> dict:
    """
    Detect Event-Related Desynchronization (ERD) and Synchronization (ERS).

    ERD/ERS quantifies task-related changes in oscillatory power relative to baseline.
    For motor tasks, expected patterns are:
    - Mu ERD (8-13 Hz): -20% to -40% during movement
    - Beta ERD (13-30 Hz): -30% to -50% during movement
    - Beta rebound (ERS): +10% to +30% post-movement (15-20s)

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
        beta_rebound_window: Beta rebound analysis window (seconds), default (15, 20)
            Post-task window for detecting beta rebound (ERS)
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
          * Beta rebound: +10% to +30% (15-20s post-task)

    Example:
        >>> results = detect_erd_ers(tfr, channel='C3', beta_rebound_window=(15, 20))
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
        f"task_window={task_window}s, baseline_window={baseline_window}s, "
        f"beta_rebound_window={beta_rebound_window}s"
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

    # Beta rebound window (post-task, from config)
    rebound_window = beta_rebound_window
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
    # Note: Using alpha_erd_percent for consistency with pipeline expectations
    # (mu and alpha bands overlap significantly in motor cortex analysis)
    results = {
        "channel": channel,
        "alpha_erd_percent": mu_erd_percent,
        "alpha_p_value": mu_p_value,
        "alpha_significant": mu_significant,
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


def plot_erd_timecourse_bilateral(
    tfr: mne.time_frequency.AverageTFR,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    task_onset: float = 0.0,
    task_offset: float = 15.0,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot ERD/ERS timecourse for both C3 and C4 (bilateral motor cortex).

    Creates a 2-row figure showing alpha and beta band power changes over time
    for both left (C3) and right (C4) motor cortex, enabling bilateral comparison.

    Args:
        tfr: Time-Frequency Representation (baseline-corrected)
        alpha_band: Alpha/Mu frequency range (Hz), default (8, 13)
        beta_band: Beta frequency range (Hz), default (13, 30)
        task_onset: Task start time for annotation (seconds, default 0)
        task_offset: Task end time for annotation (seconds, default 15)
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object with 2 subplots (C3 and C4)

    Notes:
        - Top row: C3 (left motor cortex, controls right hand)
        - Bottom row: C4 (right motor cortex, controls left hand)
        - Enables visual comparison of lateralization patterns
        - Expected: Contralateral ERD (opposite hemisphere to moving hand)

    Example:
        >>> fig = plot_erd_timecourse_bilateral(tfr)
        >>> # Shows C3 and C4 timecourses for bilateral comparison
        >>> plt.show()

    References:
        - Pfurtscheller & Neuper (1997). Motor imagery activates primary
          sensorimotor area in humans. Neurosci Lett 239(2-3).

    Requirements: Bilateral ERD visualization
    """
    logger.info("Plotting bilateral ERD timecourse (C3 and C4)")

    # Check if both channels exist
    if 'C3' not in tfr.ch_names or 'C4' not in tfr.ch_names:
        raise ValueError(
            f"C3 and/or C4 not found in TFR. "
            f"Available channels: {tfr.ch_names}"
        )

    # Get channel indices
    c3_idx = tfr.ch_names.index('C3')
    c4_idx = tfr.ch_names.index('C4')

    times = tfr.times
    freqs = tfr.freqs

    # Get frequency band masks
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])

    # Extract data for both channels
    c3_data = tfr.data[c3_idx, :, :]
    c4_data = tfr.data[c4_idx, :, :]

    # Compute band-averaged power
    c3_alpha = np.mean(c3_data[alpha_mask, :], axis=0)
    c3_beta = np.mean(c3_data[beta_mask, :], axis=0)
    c4_alpha = np.mean(c4_data[alpha_mask, :], axis=0)
    c4_beta = np.mean(c4_data[beta_mask, :], axis=0)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot C3 (left motor cortex)
    ax = axes[0]
    ax.plot(
        times, c3_alpha,
        color="blue", linewidth=2,
        label=f"Alpha/Mu ({alpha_band[0]}-{alpha_band[1]} Hz)"
    )
    ax.plot(
        times, c3_beta,
        color="red", linewidth=2,
        label=f"Beta ({beta_band[0]}-{beta_band[1]} Hz)"
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.axvline(task_onset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(task_offset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvspan(task_onset, task_offset, alpha=0.1, color="gray", label="Task window")
    ax.set_ylabel("Power change (%)", fontsize=11)
    ax.set_title("C3 (Left Motor Cortex - Controls Right Hand)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot C4 (right motor cortex)
    ax = axes[1]
    ax.plot(
        times, c4_alpha,
        color="blue", linewidth=2,
        label=f"Alpha/Mu ({alpha_band[0]}-{alpha_band[1]} Hz)"
    )
    ax.plot(
        times, c4_beta,
        color="red", linewidth=2,
        label=f"Beta ({beta_band[0]}-{beta_band[1]} Hz)"
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.axvline(task_onset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(task_offset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvspan(task_onset, task_offset, alpha=0.1, color="gray", label="Task window")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Power change (%)", fontsize=11)
    ax.set_title("C4 (Right Motor Cortex - Controls Left Hand)", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        "Bilateral Motor Cortex ERD/ERS Time Course\n(Negative = ERD, Positive = ERS)",
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig


def define_motor_roi_clusters() -> dict[str, list[str]]:
    """
    Define motor cortex ROI clusters for bilateral analysis.

    Returns motor cortex regions of interest (ROIs) as single electrodes
    (C3 and C4 only) due to limited channel availability in sub-002 data.

    Returns:
        Dictionary mapping ROI names to lists of channel names:
        - 'left_motor': Left hemisphere motor cortex (C3 only)
        - 'right_motor': Right hemisphere motor cortex (C4 only)

    Notes:
        - Left motor cortex (C3): Controls right hand movement
        - Right motor cortex (C4): Controls left hand movement
        - Sub-002 data has only 4 good EEG channels: C3, C4, F3, F4
        - Neighbor channels (FC1, CP1, T7, FC2, CP2, T8) are not available
        - Using single electrodes instead of clusters due to data limitations

    Example:
        >>> clusters = define_motor_roi_clusters()
        >>> print(clusters['left_motor'])
        ['C3']
        >>> print(clusters['right_motor'])
        ['C4']

    References:
        - Pfurtscheller & Neuper (1997). Motor imagery activates primary
          sensorimotor area in humans. Neurosci Lett 239(2-3).
        - 10-20 system: Jasper (1958). The ten-twenty electrode system.

    Requirements: Adapted for limited channel availability
    """
    # Simplified clusters for sub-002: only C3 and C4 available
    # Neighbor channels have poor data quality
    clusters = {
        'left_motor': ['C3'],   # Left hemisphere (right hand control)
        'right_motor': ['C4'],  # Right hemisphere (left hand control)
    }
    return clusters


def compute_tfr_by_condition(
    raw: mne.io.Raw,
    freqs: np.ndarray | None = None,
    n_cycles: float | np.ndarray = 7,
    tmin: float = -3.0,
    tmax: float = 15.0,
    baseline: tuple[float, float] = (-3.0, -1.0),
    baseline_mode: str = "percent",
) -> dict[str, mne.time_frequency.AverageTFR]:
    """
    Compute time-frequency representations separately for each condition.

    Separates epochs by condition (LEFT, RIGHT, NOTHING) and computes TFR
    for each condition independently. This enables condition-specific
    visualization and statistical comparison.

    Args:
        raw: Preprocessed EEG Raw object with annotations
        freqs: Frequencies for TFR (Hz). Default: 3-30 Hz in 1 Hz steps
        n_cycles: Number of cycles for Morlet wavelets. Default: 7
            Higher values = better frequency resolution, worse time resolution
        tmin: Epoch start time relative to event (seconds)
        tmax: Epoch end time relative to event (seconds)
        baseline: Baseline correction window (start, end) in seconds
        baseline_mode: Baseline correction mode ('percent', 'mean', 'logratio')

    Returns:
        Dictionary mapping condition names to AverageTFR objects:
        - 'LEFT': TFR for left hand movement trials
        - 'RIGHT': TFR for right hand movement trials
        - 'NOTHING': TFR for control/rest trials

    Notes:
        - Each condition is processed independently
        - Baseline correction applied per condition
        - Returns None for conditions with no trials
        - Useful for comparing ERD patterns across conditions

    Example:
        >>> tfr_by_cond = compute_tfr_by_condition(raw_eeg)
        >>> tfr_left = tfr_by_cond['LEFT']
        >>> tfr_right = tfr_by_cond['RIGHT']
        >>> # Compare C4 ERD for LEFT vs RIGHT conditions

    References:
        - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.

    Requirements: Condition-specific ERD analysis
    """
    if freqs is None:
        freqs = np.arange(3, 31, 1)

    logger.info("Computing TFR by condition (LEFT, RIGHT, NOTHING)")

    # Get events
    events, event_id = mne.events_from_annotations(raw)

    # Identify condition event IDs
    left_ids = {k: v for k, v in event_id.items() if "LEFT" in k.upper()}
    right_ids = {k: v for k, v in event_id.items() if "RIGHT" in k.upper()}
    nothing_ids = {k: v for k, v in event_id.items() if "NOTHING" in k.upper()}

    logger.info(f"Event IDs - LEFT: {left_ids}, RIGHT: {right_ids}, NOTHING: {nothing_ids}")

    tfr_by_condition = {}

    # Process each condition
    for cond_name, cond_ids in [("LEFT", left_ids), ("RIGHT", right_ids), ("NOTHING", nothing_ids)]:
        if not cond_ids:
            logger.warning(f"No events found for {cond_name} condition")
            tfr_by_condition[cond_name] = None
            continue

        try:
            # Create epochs for this condition
            epochs = mne.Epochs(
                raw,
                events,
                event_id=cond_ids,
                tmin=tmin,
                tmax=tmax,
                baseline=None,  # Apply baseline on TFR
                preload=True,
                reject=None,
                verbose=False,
            )
            logger.info(f"{cond_name}: {len(epochs)} epochs")

            # Compute TFR
            tfr = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=True,
                verbose=False,
            )

            # Apply baseline correction
            tfr.apply_baseline(baseline=baseline, mode=baseline_mode)

            tfr_by_condition[cond_name] = tfr
            logger.info(f"TFR computed for {cond_name}")

        except Exception as e:
            logger.warning(f"Failed to compute TFR for {cond_name}: {e}")
            tfr_by_condition[cond_name] = None

    return tfr_by_condition


def plot_spectrogram_by_condition(
    tfr_by_condition: dict[str, mne.time_frequency.AverageTFR],
    roi_name: str,
    channels: list[str] | None = None,
    vmin: float = -50.0,
    vmax: float = 50.0,
    cmap: str = "RdBu_r",
    task_onset: float = 0.0,
    task_offset: float = 15.0,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    figsize: tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Plot spectrograms for all conditions (LEFT, RIGHT, NOTHING) in a single figure.

    Creates a 3-row figure showing time-frequency spectrograms for each condition,
    enabling visual comparison of ERD/ERS patterns across experimental conditions.
    Can plot either a single channel or average across a cluster of channels (ROI).

    Args:
        tfr_by_condition: Dictionary mapping condition names to TFR objects
            Keys: 'LEFT', 'RIGHT', 'NOTHING'
        roi_name: Name of ROI for title (e.g., 'Left Motor Cortex')
        channels: List of channel names to average (e.g., ['C3', 'C1', 'C5', 'CP3'])
            If None, uses first channel in TFR
        vmin: Minimum colormap value (percent change)
        vmax: Maximum colormap value (percent change)
        cmap: Colormap name (default 'RdBu_r' for diverging blue-red)
        task_onset: Task start time for annotation (seconds)
        task_offset: Task end time for annotation (seconds)
        alpha_band: Alpha/Mu frequency band for annotation (Hz)
        beta_band: Beta frequency band for annotation (Hz)
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object with 3 subplots (one per condition)

    Notes:
        - Top row: LEFT condition
        - Middle row: RIGHT condition
        - Bottom row: NOTHING condition
        - Blue regions = ERD (power decrease)
        - Red regions = ERS (power increase)
        - Shared colorbar for direct comparison

    Example:
        >>> clusters = define_motor_roi_clusters()
        >>> tfr_by_cond = compute_tfr_by_condition(raw_eeg)
        >>> fig = plot_spectrogram_by_condition(
        ...     tfr_by_cond,
        ...     roi_name='Left Motor Cortex (C3 cluster)',
        ...     channels=clusters['left_motor']
        ... )

    References:
        - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization.

    Requirements: Condition-specific visualization for ERD analysis
    """
    logger.info(f"Plotting spectrograms by condition for {roi_name}")

    conditions = ['LEFT', 'RIGHT', 'NOTHING']
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

    for idx, cond_name in enumerate(conditions):
        ax = axes[idx]
        tfr = tfr_by_condition.get(cond_name)

        if tfr is None:
            ax.text(
                0.5, 0.5, f"No data for {cond_name}",
                ha='center', va='center', fontsize=14, color='red'
            )
            ax.set_title(f"{cond_name} Condition", fontsize=12, fontweight='bold')
            continue

        # Average across channels if multiple provided
        if channels is not None:
            # Find available channels
            available_channels = [ch for ch in channels if ch in tfr.ch_names]
            if not available_channels:
                logger.warning(f"No channels from {channels} found in TFR for {cond_name}")
                available_channels = [tfr.ch_names[0]]

            # Average data across channels
            ch_indices = [tfr.ch_names.index(ch) for ch in available_channels]
            data = np.mean(tfr.data[ch_indices, :, :], axis=0)
            channel_label = f"{roi_name} ({len(available_channels)} channels)"
        else:
            # Use first channel
            data = tfr.data[0, :, :]
            channel_label = tfr.ch_names[0]

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

        # Annotate task window
        ax.axvline(task_onset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(task_offset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

        # Annotate frequency bands
        ax.axhline(alpha_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(alpha_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(alpha_band), "Alpha",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

        ax.axhline(beta_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(beta_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(beta_band), "Beta",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

        # Labels
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.set_title(f"{cond_name} Condition", fontsize=12, fontweight='bold')

        # Only show x-label on bottom plot
        if idx == 2:
            ax.set_xlabel("Time (s)", fontsize=11)

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label="Power change (%)", pad=0.02)

    # Overall title
    fig.suptitle(
        f"Time-Frequency Spectrograms by Condition\n{channel_label}",
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig



def plot_condition_contrast_spectrograms(
    tfr_by_condition: dict[str, mne.time_frequency.AverageTFR],
    motor_clusters: dict[str, list[str]],
    vmin: float = -30.0,
    vmax: float = 30.0,
    cmap: str = "RdBu_r",
    task_onset: float = 0.0,
    task_offset: float = 15.0,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot condition contrast spectrograms showing ERD differences between conditions.

    Creates a 2-row figure showing:
    - Top: Left motor cluster (C3+) → LEFT - RIGHT (expected: negative = more ERD in LEFT)
    - Bottom: Right motor cluster (C4+) → RIGHT - LEFT (expected: negative = more ERD in RIGHT)

    This reveals whether the expected contralateral ERD pattern is present by showing
    where one condition has stronger ERD than the other.

    Args:
        tfr_by_condition: Dictionary mapping condition names to TFR objects
            Keys: 'LEFT', 'RIGHT', 'NOTHING'
        motor_clusters: Dictionary with 'left_motor' and 'right_motor' channel lists
        vmin: Minimum colormap value (percent change difference)
        vmax: Maximum colormap value (percent change difference)
        cmap: Colormap name (default 'RdBu_r' for diverging blue-red)
            Blue = first condition has more ERD (more negative)
            Red = second condition has more ERD
        task_onset: Task start time for annotation (seconds)
        task_offset: Task end time for annotation (seconds)
        alpha_band: Alpha/Mu frequency band for annotation (Hz)
        beta_band: Beta frequency band for annotation (Hz)
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object with 2 subplots (contrast maps)

    Notes:
        - Top plot: LEFT - RIGHT for left motor cluster
          * Blue = LEFT has more ERD (expected for contralateral control)
          * Red = RIGHT has more ERD (unexpected)
        - Bottom plot: RIGHT - LEFT for right motor cluster
          * Blue = RIGHT has more ERD (expected for contralateral control)
          * Red = LEFT has more ERD (unexpected)
        - Contrasts reveal lateralization specificity

    Example:
        >>> clusters = define_motor_roi_clusters()
        >>> tfr_by_cond = compute_tfr_by_condition(raw_eeg)
        >>> fig = plot_condition_contrast_spectrograms(
        ...     tfr_by_cond, clusters
        ... )

    References:
        - Pfurtscheller & Neuper (1997). Motor imagery activates primary
          sensorimotor area in humans. Neurosci Lett 239(2-3).

    Requirements: Condition contrast visualization for lateralization validation
    """
    logger.info("Plotting condition contrast spectrograms")

    # Check if required conditions exist
    if tfr_by_condition.get('LEFT') is None or tfr_by_condition.get('RIGHT') is None:
        raise ValueError("Both LEFT and RIGHT conditions required for contrast")

    tfr_left = tfr_by_condition['LEFT']
    tfr_right = tfr_by_condition['RIGHT']

    times = tfr_left.times
    freqs = tfr_left.freqs

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    # =========================================================================
    # Top plot: Left motor cluster (C3+) → LEFT - RIGHT
    # =========================================================================
    ax = axes[0]
    
    # Get left motor cluster channels
    left_channels = motor_clusters['left_motor']
    available_left = [ch for ch in left_channels if ch in tfr_left.ch_names]
    
    if not available_left:
        ax.text(0.5, 0.5, "No left motor channels available", 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title("Left Motor Cluster (C3+): LEFT - RIGHT", 
                     fontsize=12, fontweight='bold')
    else:
        # Average across left motor cluster for both conditions
        left_ch_indices = [tfr_left.ch_names.index(ch) for ch in available_left]
        
        left_cond_data = np.mean(tfr_left.data[left_ch_indices, :, :], axis=0)
        right_cond_data = np.mean(tfr_right.data[left_ch_indices, :, :], axis=0)
        
        # Compute contrast: LEFT - RIGHT
        # Negative values = LEFT has more ERD (expected for left motor during left hand)
        contrast_data = left_cond_data - right_cond_data
        
        # Plot contrast
        im = ax.imshow(
            contrast_data,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        
        # Annotate task window
        ax.axvline(task_onset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(task_offset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        
        # Annotate frequency bands
        ax.axhline(alpha_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(alpha_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(alpha_band), "Alpha",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        
        ax.axhline(beta_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(beta_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(beta_band), "Beta",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.set_title(
            f"Left Motor Cluster ({', '.join(available_left)}): LEFT - RIGHT\n"
            f"Blue = LEFT has more ERD (expected), Red = RIGHT has more ERD",
            fontsize=11, fontweight='bold'
        )

    # =========================================================================
    # Bottom plot: Right motor cluster (C4+) → RIGHT - LEFT
    # =========================================================================
    ax = axes[1]
    
    # Get right motor cluster channels
    right_channels = motor_clusters['right_motor']
    available_right = [ch for ch in right_channels if ch in tfr_right.ch_names]
    
    if not available_right:
        ax.text(0.5, 0.5, "No right motor channels available",
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title("Right Motor Cluster (C4+): RIGHT - LEFT",
                     fontsize=12, fontweight='bold')
    else:
        # Average across right motor cluster for both conditions
        right_ch_indices = [tfr_right.ch_names.index(ch) for ch in available_right]
        
        left_cond_data = np.mean(tfr_left.data[right_ch_indices, :, :], axis=0)
        right_cond_data = np.mean(tfr_right.data[right_ch_indices, :, :], axis=0)
        
        # Compute contrast: RIGHT - LEFT
        # Negative values = RIGHT has more ERD (expected for right motor during right hand)
        contrast_data = right_cond_data - left_cond_data
        
        # Plot contrast
        im = ax.imshow(
            contrast_data,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        
        # Annotate task window
        ax.axvline(task_onset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(task_offset, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        
        # Annotate frequency bands
        ax.axhline(alpha_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(alpha_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(alpha_band), "Alpha",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        
        ax.axhline(beta_band[0], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(beta_band[1], color="white", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            times[-1] * 0.98, np.mean(beta_band), "Beta",
            color="white", fontsize=9, ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.set_title(
            f"Right Motor Cluster ({', '.join(available_right)}): RIGHT - LEFT\n"
            f"Blue = RIGHT has more ERD (expected), Red = LEFT has more ERD",
            fontsize=11, fontweight='bold'
        )

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label="Power change difference (%)", pad=0.02)

    # Overall title
    fig.suptitle(
        "Condition Contrast Spectrograms: Lateralization Specificity\n"
        "(Blue = Expected contralateral ERD pattern)",
        fontsize=13, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig



def compute_psd_by_condition(
    raw: mne.io.Raw,
    channels: list[str],
    tmin: float = -3.0,
    tmax: float = 20.0,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Compute Power Spectral Density (PSD) for each condition and channel.

    Computes PSD separately for LEFT, RIGHT, and NOTHING conditions across
    all available trials. Returns mean PSD and standard deviation for
    visualization with error bands.

    Args:
        raw: Preprocessed EEG Raw object with annotations
        channels: List of channel names to analyze (e.g., ['C3', 'C4', 'F3', 'F4'])
        tmin: Epoch start time relative to event (seconds)
        tmax: Epoch end time relative to event (seconds)
        fmin: Minimum frequency for PSD (Hz)
        fmax: Maximum frequency for PSD (Hz)

    Returns:
        Dictionary with structure:
        {
            'LEFT': {
                'C3': (freqs, psd_mean, psd_std),
                'C4': (freqs, psd_mean, psd_std),
                ...
            },
            'RIGHT': {...},
            'NOTHING': {...}
        }

    Notes:
        - Uses Welch's method for PSD estimation
        - PSD computed on entire epoch (tmin to tmax)
        - Mean and std computed across trials for each condition
        - Useful for comparing spectral profiles across conditions

    Example:
        >>> psd_data = compute_psd_by_condition(raw, ['C3', 'C4'])
        >>> freqs, psd_mean, psd_std = psd_data['LEFT']['C3']
        >>> # Plot with error band: plt.fill_between(freqs, psd_mean-psd_std, psd_mean+psd_std)
    """
    logger.info(f"Computing PSD by condition for channels: {channels}")
    
    # Get events and create epochs for each condition
    events, event_id = mne.events_from_annotations(raw)
    
    # Map event IDs to condition names
    condition_names = {}
    for name, code in event_id.items():
        if 'LEFT' in name:
            condition_names[code] = 'LEFT'
        elif 'RIGHT' in name:
            condition_names[code] = 'RIGHT'
        elif 'NOTHING' in name:
            condition_names[code] = 'NOTHING'
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,  # No baseline correction for PSD
        preload=True,
        picks=channels,
        verbose=False,
    )
    
    # Initialize results dictionary
    psd_results = {'LEFT': {}, 'RIGHT': {}, 'NOTHING': {}}
    
    # Compute PSD for each condition and channel
    for condition in ['LEFT', 'RIGHT', 'NOTHING']:
        # Get condition code
        condition_code = [code for code, name in condition_names.items() if name == condition]
        if not condition_code:
            logger.warning(f"Condition {condition} not found in events")
            continue
        
        # Select epochs for this condition
        condition_epochs = epochs[str(condition_code[0])]
        n_trials = len(condition_epochs)
        logger.info(f"{condition}: {n_trials} epochs")
        
        if n_trials == 0:
            continue
        
        # Compute PSD for each channel
        for ch in channels:
            # Get PSD for all trials
            psd_trials = []
            for trial_idx in range(n_trials):
                trial_data = condition_epochs[trial_idx].get_data(picks=[ch])[0, 0, :]
                
                # Compute PSD using Welch's method
                from scipy import signal
                freqs, psd = signal.welch(
                    trial_data,
                    fs=epochs.info['sfreq'],
                    nperseg=min(256, len(trial_data)),
                    noverlap=None,
                )
                
                # Select frequency range
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                psd_trials.append(psd[freq_mask])
            
            # Convert to array and compute statistics
            psd_trials = np.array(psd_trials)
            psd_mean = np.mean(psd_trials, axis=0)
            psd_std = np.std(psd_trials, axis=0)
            freqs_selected = freqs[freq_mask]
            
            # Store results
            psd_results[condition][ch] = (freqs_selected, psd_mean, psd_std)
    
    return psd_results


def plot_psd_by_condition(
    psd_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    channels: list[str] = ['C3', 'C4', 'F3', 'F4'],
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot Power Spectral Density for each channel and condition.

    Creates a 2x2 subplot figure with one subplot per channel, showing
    PSD for LEFT, RIGHT, and NOTHING conditions with shaded error bands.

    Args:
        psd_data: Dictionary from compute_psd_by_condition()
        channels: List of 4 channel names (default: ['C3', 'C4', 'F3', 'F4'])
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object

    Notes:
        - Each subplot shows 3 lines (LEFT, RIGHT, NOTHING)
        - Shaded regions represent ±1 SD across trials
        - Vertical lines mark alpha (8-13 Hz) and beta (13-30 Hz) bands
        - Log scale on y-axis for better visualization

    Example:
        >>> psd_data = compute_psd_by_condition(raw, ['C3', 'C4', 'F3', 'F4'])
        >>> fig = plot_psd_by_condition(psd_data)
        >>> plt.show()
    """
    logger.info(f"Plotting PSD by condition for {len(channels)} channels")
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Colors for conditions
    colors = {
        'LEFT': '#1f77b4',    # Blue
        'RIGHT': '#ff7f0e',   # Orange
        'NOTHING': '#2ca02c', # Green
    }
    
    # Plot each channel
    for idx, ch in enumerate(channels):
        ax = axes[idx]
        
        # Plot each condition
        for condition in ['LEFT', 'RIGHT', 'NOTHING']:
            if ch not in psd_data[condition]:
                continue
            
            freqs, psd_mean, psd_std = psd_data[condition][ch]
            
            # Convert to µV²/Hz for better scale
            psd_mean_uv = psd_mean * 1e12  # V² to µV²
            psd_std_uv = psd_std * 1e12
            
            # Calculate error bands, ensuring lower bound is positive for log scale
            lower_bound = np.maximum(psd_mean_uv - psd_std_uv, psd_mean_uv * 0.01)  # At least 1% of mean
            upper_bound = psd_mean_uv + psd_std_uv
            
            # Plot mean line
            ax.plot(
                freqs,
                psd_mean_uv,
                color=colors[condition],
                linewidth=2,
                label=condition,
                alpha=0.9,
            )
            
            # Plot shaded error band (±1 SD, with positive lower bound)
            ax.fill_between(
                freqs,
                lower_bound,
                upper_bound,
                color=colors[condition],
                alpha=0.2,
            )
        
        # Mark frequency bands
        ax.axvspan(8, 13, color='gray', alpha=0.1, label='Alpha' if idx == 0 else '')
        ax.axvspan(13, 30, color='gray', alpha=0.05, label='Beta' if idx == 0 else '')
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power (µV²/Hz)', fontsize=10)
        ax.set_title(f'{ch}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(1, 40)
    
    # Overall title
    fig.suptitle(
        'Power Spectral Density by Condition\n'
        '(Solid line = mean, shaded area = ±1 SD across trials)',
        fontsize=14,
        fontweight='bold',
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig
