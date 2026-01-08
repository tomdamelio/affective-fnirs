"""
Multimodal Analysis Module

This module implements neurovascular coupling analysis by quantifying temporal
relationships between neural (EEG) and vascular (fNIRS) signals.

Scientific Background:
    Neurovascular coupling refers to the relationship between neural activity and
    hemodynamic responses. In motor tasks, neural desynchronization (ERD) in the
    alpha band precedes hemodynamic changes (HbO increase) by 2-5 seconds.

Key Functions:
    - extract_eeg_envelope: Extract alpha band power envelope from EEG
    - resample_to_fnirs: Resample EEG envelope to match fNIRS sampling rate
    - compute_neurovascular_coupling: Compute cross-correlation between signals
    - plot_coupling_overlay: Visualize temporal alignment of EEG and fNIRS

References:
    - Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization
      and desynchronization. Clin Neurophysiol 110(11).
    - Steinbrink et al. (2006). Simultaneous EEG-fNIRS measurements. NeuroImage 31(1).

Requirements: 7.1-7.6, 8.4
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

logger = logging.getLogger(__name__)


def extract_eeg_envelope(
    raw: mne.io.Raw,
    channel: str = "C3",
    freq_band: Tuple[float, float] = (8.0, 12.0),
    envelope_lowpass_hz: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract alpha band envelope from EEG for neurovascular coupling analysis.

    This function extracts the power envelope of oscillatory activity in a specified
    frequency band (typically alpha: 8-12 Hz) using the Hilbert transform. The
    envelope is then low-pass filtered to match the hemodynamic frequency content.

    Algorithm:
        1. Bandpass filter EEG in alpha band (8-12 Hz)
        2. Apply Hilbert transform to get analytic signal
        3. Extract envelope (magnitude of analytic signal)
        4. Low-pass filter envelope (<0.5 Hz) to match hemodynamic frequency

    Args:
        raw: MNE Raw object with EEG data (should be preprocessed: filtered, ICA)
        channel: EEG channel name (default 'C3' for motor cortex)
        freq_band: Frequency band for envelope extraction (default (8.0, 12.0) Hz for alpha)
        envelope_lowpass_hz: Low-pass cutoff for envelope smoothing (default 0.5 Hz)

    Returns:
        times: Time vector (seconds) corresponding to envelope samples
        envelope: Alpha power envelope (arbitrary units, non-negative)

    Notes:
        - Input raw should be preprocessed (filtered 1-40 Hz, ICA applied)
        - Alpha band (8-12 Hz) is typical for motor cortex analysis
        - Envelope is low-pass filtered to match hemodynamic frequency content
        - Envelope represents instantaneous power in the specified frequency band
        - For neurovascular coupling, envelope will be inverted (ERD = power decrease)

    Example:
        >>> raw_eeg = preprocess_eeg_pipeline(raw_eeg)  # Apply ICA, CAR
        >>> times, alpha_envelope = extract_eeg_envelope(raw_eeg, channel='C3')
        >>> # Envelope shape: (n_samples,), non-negative values
        >>> # For coupling: inverted_envelope = -alpha_envelope (ERD becomes positive)

    References:
        - Bruns (2004). Fourier-, Hilbert- and wavelet-based signal analysis.
          Trends in Neurosciences 27(7).
        - Requirement 7.1: Extract EEG envelope using Hilbert transform
        - Requirement 7.2: Low-pass filter envelope to match hemodynamic frequency
    """
    logger.info(
        f"Extracting alpha envelope from channel {channel}, "
        f"freq_band={freq_band} Hz, envelope_lowpass={envelope_lowpass_hz} Hz"
    )

    # Validate channel exists
    if channel not in raw.ch_names:
        available_channels = [ch for ch in raw.ch_names if "eeg" in raw.get_channel_types([ch])[0].lower()]
        raise ValueError(
            f"Channel '{channel}' not found in Raw object. "
            f"Available EEG channels: {available_channels}"
        )

    # Step 1: Bandpass filter in alpha band
    logger.debug(f"Bandpass filtering {channel} in {freq_band} Hz")
    raw_alpha = raw.copy().pick_channels([channel])
    raw_alpha.filter(
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        method="fir",
        fir_design="firwin",
        verbose=False,
    )

    # Get filtered data
    data_alpha = raw_alpha.get_data()[0]  # Shape: (n_samples,)
    times = raw_alpha.times  # Time vector in seconds

    # Step 2: Apply Hilbert transform to get analytic signal
    logger.debug("Applying Hilbert transform")
    analytic_signal = signal.hilbert(data_alpha)

    # Step 3: Extract envelope (magnitude of analytic signal)
    envelope = np.abs(analytic_signal)
    logger.debug(f"Envelope extracted, shape: {envelope.shape}, range: [{envelope.min():.2e}, {envelope.max():.2e}]")

    # Step 4: Low-pass filter envelope to match hemodynamic frequency
    logger.debug(f"Low-pass filtering envelope at {envelope_lowpass_hz} Hz")
    # Use scipy.signal.filtfilt for zero-phase filtering
    # Design Butterworth filter
    nyquist_freq = raw_alpha.info["sfreq"] / 2
    normalized_cutoff = envelope_lowpass_hz / nyquist_freq

    if normalized_cutoff >= 1.0:
        logger.warning(
            f"Envelope lowpass cutoff ({envelope_lowpass_hz} Hz) is too high "
            f"for sampling rate ({raw_alpha.info['sfreq']} Hz). Skipping envelope filtering."
        )
        envelope_filtered = envelope
    else:
        # 4th order Butterworth filter
        sos = signal.butter(4, normalized_cutoff, btype="low", output="sos")
        envelope_filtered = signal.sosfiltfilt(sos, envelope)

    logger.info(
        f"Alpha envelope extracted: {len(envelope_filtered)} samples, "
        f"duration={times[-1]:.1f}s, mean={envelope_filtered.mean():.2e}"
    )

    return times, envelope_filtered



def resample_to_fnirs(
    eeg_envelope: np.ndarray,
    eeg_times: np.ndarray,
    fnirs_times: np.ndarray,
    fnirs_sfreq: float,
) -> np.ndarray:
    """
    Resample EEG envelope to match fNIRS sampling rate for cross-correlation.

    This function resamples the EEG envelope to align with fNIRS time points,
    enabling direct comparison and cross-correlation analysis.

    Algorithm:
        1. Determine target number of samples from fNIRS time vector
        2. Use scipy.signal.resample for high-quality resampling
        3. Verify time alignment after resampling

    Args:
        eeg_envelope: EEG alpha envelope (from extract_eeg_envelope)
        eeg_times: Time vector for EEG envelope (seconds)
        fnirs_times: Time vector for fNIRS data (seconds)
        fnirs_sfreq: fNIRS sampling frequency (Hz)

    Returns:
        resampled_envelope: EEG envelope resampled to fNIRS time points

    Notes:
        - Uses scipy.signal.resample (Fourier method) for high-quality resampling
        - Alternative: mne.filter.resample (also uses Fourier method)
        - Resampling preserves envelope shape while matching fNIRS time resolution
        - Typical: EEG ~500 Hz → fNIRS ~8 Hz (downsampling by ~60x)
        - Time alignment verified by comparing duration and sample count

    Example:
        >>> eeg_times, eeg_envelope = extract_eeg_envelope(raw_eeg, channel='C3')
        >>> fnirs_times = raw_fnirs.times
        >>> fnirs_sfreq = raw_fnirs.info['sfreq']
        >>> resampled_envelope = resample_to_fnirs(eeg_envelope, eeg_times, fnirs_times, fnirs_sfreq)
        >>> # Verify: len(resampled_envelope) == len(fnirs_times)

    References:
        - scipy.signal.resample: Fourier-based resampling
        - Requirement 7.2: Resample EEG envelope to match fNIRS sampling rate
    """
    logger.info(
        f"Resampling EEG envelope from {len(eeg_envelope)} samples "
        f"to {len(fnirs_times)} samples (fNIRS rate: {fnirs_sfreq:.2f} Hz)"
    )

    # Validate input dimensions
    if len(eeg_envelope) != len(eeg_times):
        raise ValueError(
            f"EEG envelope length ({len(eeg_envelope)}) does not match "
            f"EEG times length ({len(eeg_times)})"
        )

    # Determine target number of samples
    n_samples_target = len(fnirs_times)

    # Resample using scipy.signal.resample (Fourier method)
    logger.debug(f"Resampling from {len(eeg_envelope)} to {n_samples_target} samples")
    resampled_envelope = signal.resample(eeg_envelope, n_samples_target)

    # Verify time alignment
    eeg_duration = eeg_times[-1] - eeg_times[0]
    fnirs_duration = fnirs_times[-1] - fnirs_times[0]
    duration_diff = abs(eeg_duration - fnirs_duration)

    if duration_diff > 1.0:  # More than 1 second difference
        logger.warning(
            f"Time duration mismatch: EEG={eeg_duration:.2f}s, "
            f"fNIRS={fnirs_duration:.2f}s, diff={duration_diff:.2f}s"
        )

    logger.info(
        f"Resampling complete: {len(resampled_envelope)} samples, "
        f"duration={fnirs_duration:.1f}s"
    )

    return resampled_envelope



def compute_neurovascular_coupling(
    eeg_envelope: np.ndarray,
    fnirs_hbo: np.ndarray,
    eeg_times: np.ndarray,
    fnirs_times: np.ndarray,
    fnirs_sfreq: float,
) -> Dict[str, Any]:
    """
    Compute neurovascular coupling via cross-correlation between EEG and fNIRS.

    This function quantifies the temporal relationship between neural activity
    (EEG alpha power) and hemodynamic response (HbO concentration). In motor
    tasks, alpha power decreases (ERD) precede HbO increases by 2-5 seconds.

    Algorithm:
        1. Resample EEG envelope to match fNIRS sampling rate
        2. INVERT alpha envelope (alpha decreases during activation → ERD)
        3. Compute cross-correlation with HbO time series
        4. Find lag with maximum correlation
        5. Validate negative lag (EEG precedes HbO by 2-5s)

    Args:
        eeg_envelope: Alpha power envelope from extract_eeg_envelope
        fnirs_hbo: HbO concentration time series (μM)
        eeg_times: Time vector for EEG envelope (seconds)
        fnirs_times: Time vector for fNIRS data (seconds)
        fnirs_sfreq: fNIRS sampling frequency (Hz)

    Returns:
        Dictionary with coupling metrics:
            - 'max_correlation': Peak correlation coefficient (-1 to 1)
            - 'lag_seconds': Lag at peak correlation (seconds)
            - 'lag_samples': Lag at peak correlation (samples)
            - 'lag_negative': Boolean, True if EEG precedes HbO (expected)
            - 'coupling_strength': Absolute value of max_correlation
            - 'cross_correlation': Full cross-correlation function
            - 'lags_seconds': Lag vector in seconds

    Notes:
        - Alpha envelope is INVERTED: ERD (power decrease) becomes positive signal
        - Expected pattern: Negative lag (EEG precedes HbO by 2-5 seconds)
        - Interpretation: Negative correlation between ERD and HbO increase
        - Cross-correlation computed using numpy.correlate with 'full' mode
        - Lag convention: negative lag = EEG leads, positive lag = fNIRS leads

    Example:
        >>> eeg_times, eeg_envelope = extract_eeg_envelope(raw_eeg, channel='C3')
        >>> fnirs_hbo = epochs_fnirs.average().get_data()[hbo_channel_idx]
        >>> fnirs_times = epochs_fnirs.times
        >>> coupling = compute_neurovascular_coupling(
        ...     eeg_envelope, fnirs_hbo, eeg_times, fnirs_times, fnirs_sfreq=8.12
        ... )
        >>> print(f"Max correlation: {coupling['max_correlation']:.3f}")
        >>> print(f"Lag: {coupling['lag_seconds']:.2f}s (negative={coupling['lag_negative']})")
        >>> # Expected: lag_seconds ≈ -3 to -5s, max_correlation > 0.4

    References:
        - Steinbrink et al. (2006). Simultaneous EEG-fNIRS measurements. NeuroImage 31(1).
        - Requirement 7.3: Compute cross-correlation with inverted alpha envelope
        - Requirement 7.4: Identify lag with maximum correlation
        - Requirement 7.5: Verify negative lag (EEG precedes HbO)
    """
    logger.info("Computing neurovascular coupling via cross-correlation")

    # Step 1: Resample EEG envelope to match fNIRS sampling rate
    eeg_envelope_resampled = resample_to_fnirs(
        eeg_envelope, eeg_times, fnirs_times, fnirs_sfreq
    )

    # Validate dimensions match
    if len(eeg_envelope_resampled) != len(fnirs_hbo):
        raise ValueError(
            f"Resampled EEG envelope length ({len(eeg_envelope_resampled)}) "
            f"does not match fNIRS HbO length ({len(fnirs_hbo)})"
        )

    # Step 2: INVERT alpha envelope (ERD = power decrease → becomes positive)
    logger.debug("Inverting alpha envelope (ERD becomes positive signal)")
    eeg_envelope_inverted = -eeg_envelope_resampled

    # Normalize signals for correlation (zero mean, unit variance)
    eeg_normalized = (eeg_envelope_inverted - np.mean(eeg_envelope_inverted)) / np.std(
        eeg_envelope_inverted
    )
    hbo_normalized = (fnirs_hbo - np.mean(fnirs_hbo)) / np.std(fnirs_hbo)

    # Step 3: Compute cross-correlation
    logger.debug("Computing cross-correlation")
    cross_corr = np.correlate(eeg_normalized, hbo_normalized, mode="full")

    # Normalize cross-correlation by signal lengths
    n_samples = len(eeg_normalized)
    cross_corr = cross_corr / n_samples

    # Step 4: Find lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(cross_corr))
    max_correlation = cross_corr[max_corr_idx]

    # Convert lag from samples to seconds
    # Lag convention: negative = EEG leads, positive = fNIRS leads
    lag_samples = max_corr_idx - (n_samples - 1)
    lag_seconds = lag_samples / fnirs_sfreq

    # Step 5: Validate negative lag (EEG precedes HbO)
    lag_negative = lag_seconds < 0

    # Compute lag vector for full cross-correlation function
    lags_samples = np.arange(-n_samples + 1, n_samples)
    lags_seconds = lags_samples / fnirs_sfreq

    logger.info(
        f"Neurovascular coupling computed: "
        f"max_correlation={max_correlation:.3f}, "
        f"lag={lag_seconds:.2f}s (negative={lag_negative})"
    )

    if not lag_negative:
        logger.warning(
            f"Unexpected positive lag ({lag_seconds:.2f}s): fNIRS precedes EEG. "
            f"Expected negative lag (EEG precedes fNIRS by 2-5s)."
        )
    elif lag_seconds < -10:
        logger.warning(
            f"Unusually large negative lag ({lag_seconds:.2f}s). "
            f"Expected range: -2 to -5 seconds."
        )

    return {
        "max_correlation": float(max_correlation),
        "lag_seconds": float(lag_seconds),
        "lag_samples": int(lag_samples),
        "lag_negative": bool(lag_negative),
        "coupling_strength": float(np.abs(max_correlation)),
        "cross_correlation": cross_corr,
        "lags_seconds": lags_seconds,
    }



def plot_coupling_overlay(
    eeg_envelope: np.ndarray,
    fnirs_hbo: np.ndarray,
    eeg_times: np.ndarray,
    fnirs_times: np.ndarray,
    coupling_metrics: Dict[str, Any],
    channel_eeg: str = "C3",
    channel_fnirs: str = "Motor ROI",
    task_window: Tuple[float, float] = (0.0, 15.0),
    output_path: str = None,
) -> plt.Figure:
    """
    Visualize neurovascular coupling with EEG and fNIRS time series overlay.

    This function creates a comprehensive visualization showing the temporal
    relationship between neural activity (EEG alpha envelope) and hemodynamic
    response (HbO concentration), including the optimal lag and correlation.

    Visualization Components:
        1. Top panel: Inverted alpha envelope and HbO on dual y-axes
        2. Bottom panel: Cross-correlation function with peak marked
        3. Annotations: Optimal lag, correlation strength, task window

    Args:
        eeg_envelope: Alpha power envelope (from extract_eeg_envelope)
        fnirs_hbo: HbO concentration time series (μM)
        eeg_times: Time vector for EEG envelope (seconds)
        fnirs_times: Time vector for fNIRS data (seconds)
        coupling_metrics: Output from compute_neurovascular_coupling
        channel_eeg: EEG channel name for plot title (default 'C3')
        channel_fnirs: fNIRS channel name for plot title (default 'Motor ROI')
        task_window: Task onset and offset times for shading (default (0, 15) seconds)
        output_path: Optional path to save figure (PNG format)

    Returns:
        matplotlib Figure object with coupling visualization

    Notes:
        - Alpha envelope is inverted for visualization (ERD becomes positive)
        - Dual y-axes: left for inverted alpha, right for HbO
        - Task window shaded in gray for reference
        - Cross-correlation peak marked with vertical line
        - Lag annotation shows temporal relationship

    Example:
        >>> eeg_times, eeg_envelope = extract_eeg_envelope(raw_eeg, channel='C3')
        >>> coupling = compute_neurovascular_coupling(eeg_envelope, fnirs_hbo, ...)
        >>> fig = plot_coupling_overlay(
        ...     eeg_envelope, fnirs_hbo, eeg_times, fnirs_times, coupling,
        ...     channel_eeg='C3', channel_fnirs='CCP3h-CP3 hbo'
        ... )
        >>> plt.show()

    References:
        - Requirement 7.6: Generate overlay plots showing temporal alignment
        - Requirement 8.4: Include neurovascular coupling plots in report
    """
    logger.info("Generating neurovascular coupling overlay plot")

    # Create figure with 2 subplots (time series + cross-correlation)
    fig, (ax_time, ax_xcorr) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- Top Panel: Time Series Overlay ---

    # Invert alpha envelope for visualization (ERD becomes positive)
    eeg_envelope_inverted = -eeg_envelope

    # Plot inverted alpha envelope (left y-axis)
    color_eeg = "tab:blue"
    ax_time.set_xlabel("Time (s)", fontsize=12)
    ax_time.set_ylabel(
        f"Inverted Alpha Power\n{channel_eeg} (a.u.)", color=color_eeg, fontsize=12
    )
    line_eeg = ax_time.plot(
        eeg_times, eeg_envelope_inverted, color=color_eeg, linewidth=1.5, label="EEG (inverted alpha)"
    )
    ax_time.tick_params(axis="y", labelcolor=color_eeg)
    ax_time.grid(True, alpha=0.3)

    # Plot HbO concentration (right y-axis)
    ax_hbo = ax_time.twinx()
    color_hbo = "tab:red"
    ax_hbo.set_ylabel(f"HbO Concentration\n{channel_fnirs} (μM)", color=color_hbo, fontsize=12)
    line_hbo = ax_hbo.plot(
        fnirs_times, fnirs_hbo, color=color_hbo, linewidth=1.5, label="fNIRS (HbO)"
    )
    ax_hbo.tick_params(axis="y", labelcolor=color_hbo)

    # Shade task window
    if task_window is not None:
        ax_time.axvspan(
            task_window[0],
            task_window[1],
            alpha=0.2,
            color="gray",
            label="Task window",
        )

    # Add vertical line at task onset
    ax_time.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Add title with coupling metrics
    title = (
        f"Neurovascular Coupling: {channel_eeg} ↔ {channel_fnirs}\n"
        f"Correlation: r={coupling_metrics['max_correlation']:.3f}, "
        f"Lag: {coupling_metrics['lag_seconds']:.2f}s "
        f"({'EEG leads' if coupling_metrics['lag_negative'] else 'fNIRS leads'})"
    )
    ax_time.set_title(title, fontsize=14, fontweight="bold")

    # Combine legends
    lines = line_eeg + line_hbo
    labels = [line.get_label() for line in lines]
    ax_time.legend(lines, labels, loc="upper left", fontsize=10)

    # --- Bottom Panel: Cross-Correlation Function ---

    lags_seconds = coupling_metrics["lags_seconds"]
    cross_corr = coupling_metrics["cross_correlation"]
    lag_peak = coupling_metrics["lag_seconds"]
    max_corr = coupling_metrics["max_correlation"]

    ax_xcorr.plot(lags_seconds, cross_corr, color="black", linewidth=1.5)
    ax_xcorr.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_xcorr.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Mark peak correlation
    ax_xcorr.axvline(
        lag_peak, color="red", linestyle="-", linewidth=2, label=f"Peak: {lag_peak:.2f}s"
    )
    ax_xcorr.plot(lag_peak, max_corr, "ro", markersize=10, zorder=5)

    ax_xcorr.set_xlabel("Lag (seconds)", fontsize=12)
    ax_xcorr.set_ylabel("Cross-Correlation", fontsize=12)
    ax_xcorr.set_title("Cross-Correlation Function", fontsize=12, fontweight="bold")
    ax_xcorr.grid(True, alpha=0.3)
    ax_xcorr.legend(loc="upper right", fontsize=10)

    # Annotate peak
    ax_xcorr.annotate(
        f"r={max_corr:.3f}",
        xy=(lag_peak, max_corr),
        xytext=(lag_peak + 2, max_corr + 0.1),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        logger.info(f"Saving coupling overlay plot to {output_path}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    logger.info("Coupling overlay plot generated successfully")

    return fig
