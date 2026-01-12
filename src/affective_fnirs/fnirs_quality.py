"""
fNIRS Quality Assessment Module.

This module implements quality metrics for fNIRS data following the PHOEBE framework
(Pollonini et al., 2016) and neuroimaging best practices. Quality assessment is
performed on RAW INTENSITY data before optical density conversion.

Key metrics:
- Scalp Coupling Index (SCI): Correlation between wavelengths in cardiac band
- Coefficient of Variation (CV): Noise detection in baseline periods
- Saturation Detection: ADC overflow identification
- Peak Spectral Power (PSP): Cardiac pulsation strength assessment

References:
- Pollonini et al. (2016). PHOEBE: A method for real-time mapping of optodes-scalp
  coupling in functional near-infrared spectroscopy. PMC4752525
- Hernandez & Pollonini (2020). NIRSplot: A tool for quality assessment of fNIRS
  scans. PMC7677693

Requirements: 3.1-3.10, 11.1-11.4
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import signal

logger = logging.getLogger(__name__)


def calculate_sci(
    raw: mne.io.Raw,
    freq_range: tuple[float, float] = (0.5, 2.5),
    sci_threshold: float = 0.75,
) -> dict[str, float]:
    """
    Calculate Scalp Coupling Index for each channel pair using cardiac band correlation.

    SCI measures the correlation between 760nm and 850nm wavelengths in the cardiac
    frequency band (0.5-2.5 Hz). Good optode-scalp coupling produces high correlation
    (>0.75-0.80) because cardiac pulsation affects both wavelengths similarly.

    Algorithm (following Pollonini et al., 2016):
    1. Verify Nyquist criterion: sfreq >= 2 * freq_range[1]
    2. Bandpass filter 0.5-2.5 Hz (cardiac band) using FIR filter
    3. For each source-detector pair:
       - Extract 760nm and 850nm signals
       - Compute Pearson correlation at zero lag
       - SCI = correlation coefficient (0-1)

    Args:
        raw: MNE Raw object with fNIRS data (fnirs_cw_amplitude type)
        freq_range: Bandpass filter range for cardiac isolation (default: 0.5-2.5 Hz)
        sci_threshold: Minimum acceptable SCI for logging (default 0.75)

    Returns:
        Dictionary mapping channel pair names to SCI values (0-1)
        Example: {'S1_D1': 0.85, 'S1_D2': 0.62, ...}

    Raises:
        ValueError: If sampling rate too low for cardiac band filtering
        ValueError: If channel types are not fnirs_cw_amplitude

    Notes:
        - Literature recommends SCI > 0.75-0.80 for good channels
        - Uses MNE-NIRS scalp_coupling_index() for validated calculation
        - Channels with SCI < threshold indicate poor optode-scalp coupling

    References:
        - PMC4752525: Original SCI validation
        - PMC7677693: SCI implementation in NIRSplot
    """
    # Validate channel types
    # Filter to only fNIRS channels (exclude misc, AUX, etc.)
    channel_types = raw.get_channel_types()
    fnirs_picks = [i for i, ch_type in enumerate(channel_types) if ch_type == "fnirs_cw_amplitude"]
    
    if not fnirs_picks:
        raise ValueError(
            "calculate_sci() requires at least one fnirs_cw_amplitude channel. "
            f"Found channel types: {set(channel_types)}"
        )
    
    # Create a copy with only fNIRS channels
    raw_fnirs = raw.copy().pick(fnirs_picks)
    
    logger.info(
        f"Calculating SCI on {len(fnirs_picks)} fNIRS channels "
        f"(excluded {len(channel_types) - len(fnirs_picks)} non-fNIRS channels)"
    )

    # Verify Nyquist criterion
    sfreq = raw_fnirs.info["sfreq"]
    nyquist_freq = sfreq / 2
    if freq_range[1] >= nyquist_freq:
        raise ValueError(
            f"Cardiac band high frequency ({freq_range[1]} Hz) exceeds Nyquist "
            f"frequency ({nyquist_freq} Hz) for sampling rate {sfreq} Hz. "
            f"Cannot filter in cardiac band."
        )

    logger.info(
        f"Calculating SCI with cardiac band {freq_range[0]}-{freq_range[1]} Hz "
        f"(sampling rate: {sfreq} Hz)"
    )

    # Use MNE-NIRS scalp_coupling_index for validated calculation
    try:
        from mne_nirs.preprocessing import scalp_coupling_index

        sci_raw = scalp_coupling_index(raw_fnirs)

        # Extract SCI values from the returned Raw object
        # MNE-NIRS stores SCI in the data array after computation
        sci_data = sci_raw.get_data()

        # Build dictionary mapping channel pairs to SCI values
        sci_dict = {}
        channel_names = sci_raw.ch_names

        for idx, ch_name in enumerate(channel_names):
            # Extract source-detector pair name (remove wavelength suffix)
            # Example: "S1_D1 760" -> "S1_D1"
            pair_name = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name

            # SCI is constant across time for each channel
            sci_value = float(np.mean(sci_data[idx]))

            # Store with pair name (will be same for both wavelengths)
            if pair_name not in sci_dict:
                sci_dict[pair_name] = sci_value

        # Log summary statistics
        sci_values = list(sci_dict.values())
        mean_sci = np.mean(sci_values)
        below_threshold = sum(1 for v in sci_values if v < sci_threshold)

        logger.info(
            f"SCI calculation complete: mean={mean_sci:.3f}, "
            f"{below_threshold}/{len(sci_values)} pairs below threshold ({sci_threshold})"
        )

        return sci_dict

    except ImportError:
        logger.warning(
            "mne_nirs.preprocessing.scalp_coupling_index not available. "
            "Falling back to manual implementation."
        )

        # Manual implementation as fallback
        return _calculate_sci_manual(raw_fnirs, freq_range, sci_threshold)


def _calculate_sci_manual(
    raw: mne.io.Raw,
    freq_range: tuple[float, float],
    sci_threshold: float,
) -> dict[str, float]:
    """
    Manual SCI calculation as fallback when MNE-NIRS not available.

    Args:
        raw: MNE Raw object with fNIRS data
        freq_range: Cardiac band frequency range
        sci_threshold: Threshold for logging

    Returns:
        Dictionary mapping channel pairs to SCI values
    """
    # Filter in cardiac band
    raw_filtered = raw.copy().filter(
        l_freq=freq_range[0],
        h_freq=freq_range[1],
        method="fir",
        verbose=False,
    )

    # Get filtered data
    data_filtered = raw_filtered.get_data()
    channel_names = raw_filtered.ch_names

    # Group channels by source-detector pair
    pairs = {}
    for idx, ch_name in enumerate(channel_names):
        # Extract pair name and wavelength
        if " " in ch_name:
            pair_name, wavelength = ch_name.rsplit(" ", 1)
        else:
            # Fallback: assume channel name is the pair
            pair_name = ch_name
            wavelength = "unknown"

        if pair_name not in pairs:
            pairs[pair_name] = {}
        pairs[pair_name][wavelength] = data_filtered[idx]

    # Calculate correlation for each pair
    sci_dict = {}
    for pair_name, wavelengths in pairs.items():
        if "760" in wavelengths and "850" in wavelengths:
            # Compute Pearson correlation at zero lag
            signal_760 = wavelengths["760"]
            signal_850 = wavelengths["850"]

            # Pearson correlation coefficient
            correlation = np.corrcoef(signal_760, signal_850)[0, 1]
            sci_dict[pair_name] = float(correlation)
        else:
            logger.warning(
                f"Pair {pair_name} missing wavelengths: {list(wavelengths.keys())}"
            )

    # Log summary
    sci_values = list(sci_dict.values())
    mean_sci = np.mean(sci_values)
    below_threshold = sum(1 for v in sci_values if v < sci_threshold)

    logger.info(
        f"Manual SCI calculation complete: mean={mean_sci:.3f}, "
        f"{below_threshold}/{len(sci_values)} pairs below threshold ({sci_threshold})"
    )

    return sci_dict




def calculate_coefficient_of_variation(
    raw: mne.io.Raw,
    baseline_annotations: list[tuple[float, float]] | None = None,
    cv_threshold: float = 15.0,
) -> dict[str, float]:
    """
    Calculate Coefficient of Variation during baseline periods only.

    CV measures signal stability by computing (std / mean) * 100%. High CV indicates
    noisy or unstable signals. CRITICAL: Must be calculated on RAW INTENSITY data,
    NOT optical density (OD has mean ≈ 0, which distorts CV calculation).

    Algorithm:
    1. Extract baseline segments (e.g., 5s before each task block)
    2. For each channel in RAW INTENSITY:
       - Compute mean and std across baseline samples
       - CV = (std / mean) * 100%
    3. High CV indicates noisy/unstable signal

    Args:
        raw: MNE Raw object with fNIRS data (raw intensity, before OD conversion)
        baseline_annotations: List of (start_time, end_time) tuples for baseline periods.
            If None, uses entire recording (not recommended for task data).
        cv_threshold: Maximum acceptable CV percentage (default 15%)

    Returns:
        Dictionary mapping channel names to CV percentages
        Example: {'S1_D1 760': 8.2, 'S1_D1 850': 12.5, ...}

    Raises:
        ValueError: If channel types are not fnirs_cw_amplitude
        ValueError: If baseline_annotations is empty

    Notes:
        - Typical thresholds: 7.5-15% depending on study requirements
        - Movement artifacts increase std → higher CV
        - Restricting to baseline avoids task-related variance (Req. 3.8)
        - CV is informative but may not be strict exclusion criterion if SCI/PSP pass

    References:
        - PMC7677693: CV calculation in baseline periods
        - PMC4752525: CV as complementary quality metric
    """
    # Filter to only fNIRS channels (exclude AUX/misc channels)
    channel_types = raw.get_channel_types()
    fnirs_channels = [
        ch
        for ch, ch_type in zip(raw.ch_names, channel_types)
        if ch_type == "fnirs_cw_amplitude"
    ]

    if len(fnirs_channels) == 0:
        raise ValueError(
            "No fnirs_cw_amplitude channels found. "
            f"Available channel types: {set(channel_types)}"
        )

    # Pick only fNIRS channels
    raw_fnirs = raw.copy().pick(fnirs_channels)

    # Get data and channel names
    data = raw_fnirs.get_data()
    channel_names = raw_fnirs.ch_names
    sfreq = raw_fnirs.info["sfreq"]

    # Determine baseline segments
    if baseline_annotations is None:
        logger.warning(
            "No baseline annotations provided. Using entire recording for CV "
            "calculation (not recommended for task data)."
        )
        # Use entire recording
        baseline_data = data
    else:
        if len(baseline_annotations) == 0:
            raise ValueError("baseline_annotations is empty. Provide baseline periods.")

        # Extract baseline segments
        baseline_segments = []
        for start_time, end_time in baseline_annotations:
            # Convert times to sample indices
            start_sample = int(start_time * sfreq)
            end_sample = int(end_time * sfreq)

            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(data.shape[1], end_sample)

            if start_sample < end_sample:
                baseline_segments.append(data[:, start_sample:end_sample])

        if len(baseline_segments) == 0:
            raise ValueError(
                "No valid baseline segments found. Check baseline_annotations."
            )

        # Concatenate all baseline segments
        baseline_data = np.concatenate(baseline_segments, axis=1)

    # Calculate CV for each channel
    cv_dict = {}
    for idx, ch_name in enumerate(channel_names):
        channel_baseline = baseline_data[idx]

        # Compute mean and std
        mean_intensity = np.mean(channel_baseline)
        std_intensity = np.std(channel_baseline, ddof=1)

        # Calculate CV as percentage
        if mean_intensity != 0:
            cv_percent = (std_intensity / mean_intensity) * 100.0
        else:
            # Mean is zero (disconnected or saturated)
            cv_percent = np.inf
            logger.warning(
                f"Channel {ch_name} has zero mean intensity. Setting CV to inf."
            )

        cv_dict[ch_name] = float(cv_percent)

    # Log summary statistics
    cv_values = [v for v in cv_dict.values() if not np.isinf(v)]
    if cv_values:
        mean_cv = np.mean(cv_values)
        above_threshold = sum(1 for v in cv_values if v > cv_threshold)

        logger.info(
            f"CV calculation complete: mean={mean_cv:.2f}%, "
            f"{above_threshold}/{len(cv_values)} channels above threshold ({cv_threshold}%)"
        )
    else:
        logger.warning("All channels have infinite CV (zero mean intensity).")

    return cv_dict




def detect_saturation(
    raw: mne.io.Raw,
    adc_max: float | None = None,
    saturation_threshold: float = 0.95,
    max_saturation_percent: float = 5.0,
) -> dict[str, float]:
    """
    Detect signal saturation in raw intensity data.

    Saturation occurs when light intensity exceeds the ADC (Analog-to-Digital Converter)
    maximum range, causing clipping. Saturated channels produce unreliable data.

    Algorithm:
    1. Determine ADC maximum:
       - If provided: use adc_max parameter
       - If unknown: estimate as max(data) * 1.05 (assumes some headroom)
       - For 16-bit ADC: typically 65,535
    2. For each channel:
       - Count samples where intensity > saturation_threshold * adc_max
       - Saturation % = (saturated_samples / total_samples) * 100
    3. Mark channel as bad if saturation % > max_saturation_percent

    Args:
        raw: MNE Raw object with fNIRS data (raw intensity)
        adc_max: Maximum ADC value (if None, estimated from data)
        saturation_threshold: Fraction of ADC range (default 0.95 = 95%)
        max_saturation_percent: Maximum allowed saturation percentage (default 5%)

    Returns:
        Dictionary mapping channel names to saturation percentages
        Example: {'S1_D1 760': 0.2, 'S1_D1 850': 8.5, ...}

    Notes:
        - Saturation indicates optode too close to scalp or excessive light
        - Saturated channels produce clipped, unreliable data
        - Both thresholds are configurable via QualityThresholds dataclass

    Example:
        If adc_max=65535 and saturation_threshold=0.95:
        Samples > 62,258 are considered saturated
    """
    # Filter to only fNIRS channels (exclude AUX/misc channels)
    channel_types = raw.get_channel_types()
    fnirs_channels = [
        ch
        for ch, ch_type in zip(raw.ch_names, channel_types)
        if ch_type == "fnirs_cw_amplitude"
    ]

    if len(fnirs_channels) == 0:
        raise ValueError(
            "No fnirs_cw_amplitude channels found. "
            f"Available channel types: {set(channel_types)}"
        )

    # Pick only fNIRS channels
    raw_fnirs = raw.copy().pick(fnirs_channels)

    # Get data and channel names
    data = raw_fnirs.get_data()
    channel_names = raw_fnirs.ch_names

    # Determine ADC maximum
    if adc_max is None:
        # Estimate from data (assume max value has 5% headroom)
        data_max = np.max(data)
        adc_max = data_max * 1.05
        logger.info(
            f"ADC maximum not provided. Estimated from data: {adc_max:.1f} "
            f"(data max: {data_max:.1f})"
        )
    else:
        logger.info(f"Using provided ADC maximum: {adc_max}")

    # Calculate saturation threshold value
    saturation_value = saturation_threshold * adc_max

    # Calculate saturation percentage for each channel
    saturation_dict = {}
    total_samples = data.shape[1]

    for idx, ch_name in enumerate(channel_names):
        channel_data = data[idx]

        # Count saturated samples
        saturated_samples = np.sum(channel_data > saturation_value)
        saturation_percent = (saturated_samples / total_samples) * 100.0

        saturation_dict[ch_name] = float(saturation_percent)

    # Log summary statistics
    saturation_values = list(saturation_dict.values())
    mean_saturation = np.mean(saturation_values)
    above_threshold = sum(1 for v in saturation_values if v > max_saturation_percent)

    logger.info(
        f"Saturation detection complete: mean={mean_saturation:.2f}%, "
        f"{above_threshold}/{len(saturation_values)} channels above threshold "
        f"({max_saturation_percent}%)"
    )

    return saturation_dict


def detect_flat_signal(
    raw: mne.io.Raw,
    variance_threshold: float = 1e-6,
) -> dict[str, bool]:
    """
    Detect flat signals indicating contact loss or disconnected optodes.

    Flat signals have near-zero variance, indicating no physiological signal is
    being recorded (disconnected electrode, poor contact, or hardware failure).

    Algorithm:
    1. For each channel:
       - Compute variance across all samples
       - If variance < threshold, mark as flat
    2. Return boolean dict indicating flat channels

    Args:
        raw: MNE Raw object with fNIRS data
        variance_threshold: Minimum variance for valid signal (default 1e-6)

    Returns:
        Dictionary mapping channel names to boolean (True = flat signal)
        Example: {'S1_D1 760': False, 'S1_D1 850': True, ...}

    Notes:
        - Flat signals indicate hardware or contact issues
        - Threshold should be adjusted based on signal units and expected range
        - Flat channels should be marked as BAD
    """
    # Get data and channel names
    data = raw.get_data()
    channel_names = raw.ch_names

    # Calculate variance for each channel
    flat_dict = {}
    for idx, ch_name in enumerate(channel_names):
        channel_data = data[idx]

        # Compute variance
        variance = np.var(channel_data, ddof=1)

        # Check if flat
        is_flat = variance < variance_threshold
        flat_dict[ch_name] = bool(is_flat)

    # Log summary
    flat_count = sum(flat_dict.values())
    logger.info(
        f"Flat signal detection complete: {flat_count}/{len(flat_dict)} channels flat "
        f"(variance < {variance_threshold})"
    )

    return flat_dict




def assess_cardiac_power(
    raw: mne.io.Raw,
    freq_range: tuple[float, float] = (0.5, 2.5),
    power_threshold: float = 0.1,
) -> dict[str, float]:
    """
    Compute Peak Spectral Power (PSP) in cardiac band following PHOEBE framework.

    PSP quantifies the strength of cardiac pulsation in the fNIRS signal. Clear
    cardiac pulsation (PSP > 0.1) indicates good optode-scalp coupling. Absence
    of cardiac peak suggests poor contact.

    Algorithm (Pollonini et al., 2016):
    1. Bandpass filter 0.5-2.5 Hz (cardiac band)
    2. Compute Power Spectral Density using Welch's method:
       - Window: Hamming, length = 2 * sfreq (2-second windows)
       - Overlap: 50%
    3. Identify peak power within 0.5-2.5 Hz (typically ~1 Hz for heart rate)
    4. Normalize peak power:
       - PSP = peak_power / total_power_in_band
    5. PSP > 0.1 indicates clear cardiac pulsation (good coupling)

    Args:
        raw: MNE Raw object with fNIRS data (raw intensity)
        freq_range: Cardiac frequency band (default: 0.5-2.5 Hz)
        power_threshold: Minimum normalized PSP (default 0.1 per PHOEBE)

    Returns:
        Dictionary mapping channel names to normalized PSP values
        Example: {'S1_D1 760': 0.25, 'S1_D1 850': 0.08, ...}

    Raises:
        ValueError: If sampling rate too low for cardiac band analysis

    Notes:
        - PSP complements SCI: both assess cardiac pulsation quality
        - PSP < 0.1 suggests weak/absent pulse → poor coupling
        - Threshold 0.1 is empirically validated in PHOEBE framework

    References:
        - PMC4752525: PSP definition and validation
        - artinis.com: PSP implementation guidelines
    """
    # Filter to only fNIRS channels (exclude AUX/misc channels)
    channel_types = raw.get_channel_types()
    fnirs_channels = [
        ch
        for ch, ch_type in zip(raw.ch_names, channel_types)
        if ch_type == "fnirs_cw_amplitude"
    ]

    if len(fnirs_channels) == 0:
        raise ValueError(
            "No fnirs_cw_amplitude channels found. "
            f"Available channel types: {set(channel_types)}"
        )

    # Pick only fNIRS channels
    raw_fnirs = raw.copy().pick(fnirs_channels)

    # Validate sampling rate
    sfreq = raw_fnirs.info["sfreq"]
    nyquist_freq = sfreq / 2
    if freq_range[1] >= nyquist_freq:
        raise ValueError(
            f"Cardiac band high frequency ({freq_range[1]} Hz) exceeds Nyquist "
            f"frequency ({nyquist_freq} Hz) for sampling rate {sfreq} Hz."
        )

    # Get data and channel names
    data = raw_fnirs.get_data()
    channel_names = raw_fnirs.ch_names

    logger.info(
        f"Assessing cardiac power in {freq_range[0]}-{freq_range[1]} Hz band "
        f"(sampling rate: {sfreq} Hz)"
    )

    # Calculate PSP for each channel
    psp_dict = {}

    # Welch parameters
    nperseg = int(2 * sfreq)  # 2-second windows
    noverlap = nperseg // 2  # 50% overlap

    for idx, ch_name in enumerate(channel_names):
        channel_data = data[idx]

        # Compute Power Spectral Density using Welch's method
        freqs, psd = signal.welch(
            channel_data,
            fs=sfreq,
            window="hamming",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        # Find indices corresponding to cardiac band
        cardiac_band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        cardiac_freqs = freqs[cardiac_band_mask]
        cardiac_psd = psd[cardiac_band_mask]

        if len(cardiac_psd) == 0:
            logger.warning(
                f"Channel {ch_name}: No frequencies in cardiac band. Setting PSP=0."
            )
            psp_dict[ch_name] = 0.0
            continue

        # Find peak power in cardiac band
        peak_power = np.max(cardiac_psd)

        # Calculate total power in cardiac band (integrate PSD)
        total_power = np.trapz(cardiac_psd, cardiac_freqs)

        # Normalize: PSP = peak_power / total_power
        if total_power > 0:
            psp_normalized = peak_power / total_power
        else:
            psp_normalized = 0.0
            logger.warning(
                f"Channel {ch_name}: Zero total power in cardiac band. Setting PSP=0."
            )

        psp_dict[ch_name] = float(psp_normalized)

    # Log summary statistics
    psp_values = list(psp_dict.values())
    mean_psp = np.mean(psp_values)
    below_threshold = sum(1 for v in psp_values if v < power_threshold)

    logger.info(
        f"Cardiac power assessment complete: mean PSP={mean_psp:.3f}, "
        f"{below_threshold}/{len(psp_values)} channels below threshold ({power_threshold})"
    )

    return psp_dict




def mark_bad_channels(
    raw: mne.io.Raw,
    sci_values: dict[str, float],
    saturation_values: dict[str, float],
    cardiac_power: dict[str, float],
    cv_values: dict[str, float] | None = None,
    flat_signals: dict[str, bool] | None = None,
    sci_threshold: float = 0.75,
    saturation_threshold: float = 5.0,
    psp_threshold: float = 0.1,
    cv_threshold: float = 15.0,
) -> tuple[mne.io.Raw, dict[str, list[str]]]:
    """
    Mark channels as BAD based on multiple quality metrics.

    A channel is marked as BAD if ANY of the following conditions are met:
    1. SCI < sci_threshold (poor optode-scalp coupling)
    2. Saturation % > saturation_threshold (ADC overflow)
    3. PSP < psp_threshold (weak cardiac pulsation)
    4. CV > cv_threshold (excessive noise in baseline) [optional]
    5. Flat signal detected (near-zero variance) [optional]

    Algorithm:
    1. For each channel, evaluate all quality criteria
    2. If any criterion fails, add channel to raw.info['bads']
    3. Record the reason(s) for marking as bad
    4. If ALL long channels are marked bad, emit warning (Req. 11.4)

    Args:
        raw: MNE Raw object with fNIRS data
        sci_values: SCI for each channel pair (pair name → SCI)
        saturation_values: Saturation percentage for each channel
        cardiac_power: PSP for each channel
        cv_values: CV for each channel (optional, informative)
        flat_signals: Flat signal detection for each channel (optional)
        sci_threshold: Minimum acceptable SCI (default 0.75)
        saturation_threshold: Maximum acceptable saturation % (default 5%)
        psp_threshold: Minimum acceptable PSP (default 0.1)
        cv_threshold: Maximum acceptable CV % (default 15%)

    Returns:
        Tuple of:
        - Raw object with bad channels marked in raw.info['bads']
        - Dictionary mapping channel names to list of failure reasons

    Notes:
        - All thresholds are configurable via QualityThresholds dataclass
        - CV is typically informative rather than strict exclusion criterion
        - Short channels (<15mm) may have different quality profiles
        - Quality report should list all metrics for transparency (Req. 3.9, 8.1)

    Example bad channel entry:
        raw.info['bads'] = ['S1_D1 760', 'S1_D1 850']
        reasons = {
            'S1_D1 760': ['Low SCI (0.45 < 0.75)', 'High saturation (8.2% > 5%)'],
            'S1_D1 850': ['Low SCI (0.45 < 0.75)', 'High saturation (8.2% > 5%)']
        }
    """
    channel_names = raw.ch_names
    bad_channels = []
    failure_reasons = {}

    logger.info(
        f"Marking bad channels with thresholds: SCI>{sci_threshold}, "
        f"Saturation<{saturation_threshold}%, PSP>{psp_threshold}, CV<{cv_threshold}%"
    )

    for ch_name in channel_names:
        reasons = []

        # Extract pair name for SCI lookup
        pair_name = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name

        # Check SCI (pair-level metric)
        if pair_name in sci_values:
            sci = sci_values[pair_name]
            if sci < sci_threshold:
                reasons.append(f"Low SCI ({sci:.2f} < {sci_threshold})")

        # Check saturation (channel-level metric)
        if ch_name in saturation_values:
            saturation = saturation_values[ch_name]
            if saturation > saturation_threshold:
                reasons.append(
                    f"High saturation ({saturation:.1f}% > {saturation_threshold}%)"
                )

        # Check cardiac power (channel-level metric)
        if ch_name in cardiac_power:
            psp = cardiac_power[ch_name]
            if psp < psp_threshold:
                reasons.append(f"Low cardiac power ({psp:.2f} < {psp_threshold})")

        # Check CV (optional, informative)
        if cv_values is not None and ch_name in cv_values:
            cv = cv_values[ch_name]
            if not np.isinf(cv) and cv > cv_threshold:
                reasons.append(f"High CV ({cv:.1f}% > {cv_threshold}%)")

        # Check flat signal (optional)
        if flat_signals is not None and ch_name in flat_signals:
            if flat_signals[ch_name]:
                reasons.append("Flat signal (near-zero variance)")

        # Mark as bad if any criterion failed
        if reasons:
            bad_channels.append(ch_name)
            failure_reasons[ch_name] = reasons

    # Update raw.info['bads']
    raw.info["bads"] = bad_channels

    # Log summary
    logger.info(
        f"Marked {len(bad_channels)}/{len(channel_names)} channels as BAD "
        f"({len(bad_channels)/len(channel_names)*100:.1f}%)"
    )

    # Check if ALL long channels are marked bad (Req. 11.4)
    # Identify long channels (source-detector distance >= 15mm)
    # For now, assume channels without "short" in name are long channels
    long_channels = [
        ch for ch in channel_names if "short" not in ch.lower() and ch in bad_channels
    ]

    if len(long_channels) == len(channel_names):
        logger.warning(
            "⚠️  ALL channels marked as bad! Consider adjusting quality thresholds:\n"
            f"  - Decrease SCI threshold (current: {sci_threshold})\n"
            f"  - Increase saturation threshold (current: {saturation_threshold}%)\n"
            f"  - Decrease PSP threshold (current: {psp_threshold})\n"
            f"  - Increase CV threshold (current: {cv_threshold}%)\n"
            "  Or check optode contact quality during recording."
        )

    # Log detailed failure reasons
    for ch_name, reasons in failure_reasons.items():
        logger.debug(f"Channel {ch_name} marked BAD: {'; '.join(reasons)}")

    return raw, failure_reasons




def generate_quality_heatmap(
    raw: mne.io.Raw,
    sci_values: dict[str, float],
    saturation_values: dict[str, float],
    cardiac_power: dict[str, float],
    cv_values: dict[str, float] | None = None,
    failure_reasons: dict[str, list[str]] | None = None,
    output_path: Path | None = None,
) -> Figure:
    """
    Generate spatial heatmap showing good/bad channel distribution.

    Visualization helps identify regions with poor optode contact (e.g., hair
    interference). Creates a color-coded plot with quality metrics.

    Algorithm:
    1. Extract channel names and quality status
    2. Create bar plot or table visualization:
       - Color code by SCI value: green (>0.8), yellow (0.6-0.8), red (<0.6)
       - Annotate bad channels with failure reasons
    3. Include all quality metrics in visualization

    Args:
        raw: MNE Raw object with fNIRS data
        sci_values: SCI for each channel pair
        saturation_values: Saturation percentage for each channel
        cardiac_power: PSP for each channel
        cv_values: CV for each channel (optional)
        failure_reasons: Dictionary of failure reasons per channel
        output_path: Optional save path for figure

    Returns:
        Matplotlib figure with quality visualization

    Notes:
        - Spatial 3D visualization requires montage coordinates
        - Fallback to table/bar plot if coordinates unavailable
        - Helps identify systematic issues (e.g., poor contact in frontal region)
        - Validates Req. 3.10: spatial quality visualization
    """
    channel_names = raw.ch_names
    bad_channels = raw.info["bads"]

    # Create figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("fNIRS Quality Assessment", fontsize=16, fontweight="bold")

    # Subplot 1: SCI values
    ax1 = axes[0, 0]
    pair_names = list(sci_values.keys())
    sci_vals = list(sci_values.values())

    # Color code by SCI value
    colors = []
    for sci in sci_vals:
        if sci > 0.8:
            colors.append("green")
        elif sci > 0.6:
            colors.append("yellow")
        else:
            colors.append("red")

    ax1.barh(pair_names, sci_vals, color=colors, alpha=0.7)
    ax1.axvline(x=0.75, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax1.set_xlabel("Scalp Coupling Index (SCI)")
    ax1.set_title("SCI per Channel Pair")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    # Subplot 2: Saturation percentages
    ax2 = axes[0, 1]
    sat_vals = [saturation_values.get(ch, 0) for ch in channel_names]
    colors_sat = ["red" if v > 5.0 else "green" for v in sat_vals]

    # Show ALL channels (removed 20-channel limit)
    display_channels = channel_names
    display_sat = sat_vals
    display_colors = colors_sat

    ax2.barh(display_channels, display_sat, color=display_colors, alpha=0.7)
    ax2.axvline(x=5.0, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax2.set_xlabel("Saturation (%)")
    ax2.set_title(f"Saturation per Channel (showing {len(display_channels)})")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    # Subplot 3: Cardiac power (PSP)
    ax3 = axes[1, 0]
    psp_vals = [cardiac_power.get(ch, 0) for ch in channel_names]
    colors_psp = ["green" if v > 0.1 else "red" for v in psp_vals]

    # Show ALL channels (removed 20-channel limit)
    display_psp = psp_vals
    display_colors_psp = colors_psp

    ax3.barh(display_channels, display_psp, color=display_colors_psp, alpha=0.7)
    ax3.axvline(x=0.1, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax3.set_xlabel("Peak Spectral Power (PSP)")
    ax3.set_title(f"Cardiac Power per Channel (showing {len(display_channels)})")
    ax3.legend()
    ax3.grid(axis="x", alpha=0.3)

    # Subplot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate summary statistics
    total_channels = len(channel_names)
    bad_count = len(bad_channels)
    good_count = total_channels - bad_count

    mean_sci = np.mean(list(sci_values.values()))
    mean_sat = np.mean(sat_vals)
    mean_psp = np.mean(psp_vals)

    summary_text = f"""
    Quality Summary
    ═══════════════════════════════
    Total Channels: {total_channels}
    Good Channels: {good_count} ({good_count/total_channels*100:.1f}%)
    Bad Channels: {bad_count} ({bad_count/total_channels*100:.1f}%)
    
    Mean Metrics:
    ───────────────────────────────
    SCI: {mean_sci:.3f}
    Saturation: {mean_sat:.2f}%
    Cardiac Power: {mean_psp:.3f}
    """

    if cv_values is not None:
        cv_vals = [v for v in cv_values.values() if not np.isinf(v)]
        if cv_vals:
            mean_cv = np.mean(cv_vals)
            summary_text += f"    CV: {mean_cv:.2f}%\n"

    # Add bad channel list
    if bad_channels:
        summary_text += f"\n    Bad Channels ({len(bad_channels)}):\n"
        summary_text += "    ───────────────────────────────\n"
        for ch in bad_channels[:10]:  # Show first 10
            summary_text += f"    • {ch}\n"
            if failure_reasons and ch in failure_reasons:
                for reason in failure_reasons[ch][:2]:  # Show first 2 reasons
                    summary_text += f"      - {reason}\n"
        if len(bad_channels) > 10:
            summary_text += f"    ... and {len(bad_channels) - 10} more\n"

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Quality heatmap saved to: {output_path}")

    return fig


def generate_quality_summary_table(
    raw: mne.io.Raw,
    sci_values: dict[str, float],
    saturation_values: dict[str, float],
    cardiac_power: dict[str, float],
    cv_values: dict[str, float] | None = None,
    failure_reasons: dict[str, list[str]] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate quality summary table (TSV) with all metrics per channel.

    Creates a comprehensive table with all quality metrics for each channel,
    suitable for BIDS derivatives output.

    Args:
        raw: MNE Raw object with fNIRS data
        sci_values: SCI for each channel pair
        saturation_values: Saturation percentage for each channel
        cardiac_power: PSP for each channel
        cv_values: CV for each channel (optional)
        failure_reasons: Dictionary of failure reasons per channel
        output_path: Optional save path for TSV file

    Returns:
        Pandas DataFrame with quality metrics

    Notes:
        - Output format: BIDS-compliant TSV with JSON data dictionary
        - Columns: channel_name, sci, saturation_percent, cardiac_power, cv, is_bad, reason
        - Validates Req. 3.9: quality report generation
    """
    channel_names = raw.ch_names
    bad_channels = raw.info["bads"]

    # Build table data
    table_data = []
    for ch_name in channel_names:
        # Extract pair name for SCI lookup
        pair_name = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name

        row = {
            "channel_name": ch_name,
            "sci": sci_values.get(pair_name, np.nan),
            "saturation_percent": saturation_values.get(ch_name, np.nan),
            "cardiac_power": cardiac_power.get(ch_name, np.nan),
            "is_bad": ch_name in bad_channels,
        }

        # Add CV if available
        if cv_values is not None:
            row["cv_percent"] = cv_values.get(ch_name, np.nan)

        # Add failure reasons
        if failure_reasons and ch_name in failure_reasons:
            row["reason"] = "; ".join(failure_reasons[ch_name])
        else:
            row["reason"] = "" if ch_name not in bad_channels else "Unknown"

        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Quality summary table saved to: {output_path}")

        # Create JSON data dictionary
        json_path = output_path.with_suffix(".json")
        data_dict = {
            "channel_name": {
                "Description": "fNIRS channel name",
                "Units": "n/a",
            },
            "sci": {
                "Description": "Scalp Coupling Index (correlation between wavelengths in cardiac band)",
                "Units": "unitless",
                "Range": [0, 1],
                "Threshold": 0.75,
            },
            "saturation_percent": {
                "Description": "Percentage of samples exceeding ADC saturation threshold",
                "Units": "percent",
                "Range": [0, 100],
                "Threshold": 5.0,
            },
            "cardiac_power": {
                "Description": "Peak Spectral Power in cardiac band (0.5-2.5 Hz)",
                "Units": "unitless",
                "Range": [0, 1],
                "Threshold": 0.1,
            },
            "is_bad": {
                "Description": "Boolean indicating if channel marked as bad",
                "Levels": {"True": "Bad channel", "False": "Good channel"},
            },
            "reason": {
                "Description": "Semicolon-separated list of failure reasons for bad channels",
                "Units": "n/a",
            },
        }

        if cv_values is not None:
            data_dict["cv_percent"] = {
                "Description": "Coefficient of Variation in baseline periods",
                "Units": "percent",
                "Range": [0, 100],
                "Threshold": 15.0,
            }

        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
        logger.info(f"Data dictionary saved to: {json_path}")

    return df


