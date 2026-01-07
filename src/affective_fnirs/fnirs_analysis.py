"""
fNIRS Analysis Module for Hemodynamic Response Function (HRF) Extraction.

This module provides functions for extracting and validating hemodynamic responses
from fNIRS data during motor tasks. It implements HRF temporal dynamics validation,
quality metrics computation, and visualization following neuroscience best practices.

Key Functions:
    - create_fnirs_epochs: Extract fNIRS epochs with extended window for HRF
    - identify_motor_roi_channel: Find fNIRS channel closest to motor cortex (C3)
    - extract_hrf: Extract averaged HRF for a channel
    - validate_hrf_temporal_dynamics: Validate HRF onset, peak, and plateau
    - compute_hrf_quality_metrics: Compute SNR and trial consistency
    - plot_hrf_curves: Visualize HRF with annotations
    - plot_hrf_spatial_map: Spatial map of HRF amplitude

Scientific Background:
    The Hemodynamic Response Function (HRF) reflects neurovascular coupling:
    neural activity triggers increased cerebral blood flow, causing measurable
    changes in oxygenated (HbO) and deoxygenated (HbR) hemoglobin.

    Typical HRF Characteristics for Motor Tasks:
    - Onset latency: HbO begins rising ~2-3s after stimulus onset
    - Time-to-peak: Peak/plateau occurs ~5-8s post-stimulus
    - Plateau for sustained tasks: 15s tapping → sustained elevation (5-15s)
    - Return to baseline: HbO returns to baseline ~20-30s after task cessation
    - HbR inverse pattern: Typically decreases (opposite to HbO)

References:
    - Obrig & Villringer (2003). Beyond the visible—imaging the human brain with light.
      J Cereb Blood Flow Metab 23(1).
    - Scholkmann et al. (2014). A review on continuous wave fNIRS. NeuroImage 85.
    - Pinti et al. (2018). The present and future use of fNIRS. Ann NY Acad Sci 1464(1).

Requirements: 6.5-6.12, 8.3
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def create_fnirs_epochs(
    raw: mne.io.Raw,
    event_id: dict[str, int],
    tmin: float = -5.0,
    tmax: float = 30.0,
    baseline: tuple[float, float] = (-5.0, 0.0),
) -> mne.Epochs:
    """
    Extract fNIRS epochs with extended window for HRF return to baseline.

    Epoch window captures:
    - Pre-stimulus baseline: -5 to 0s (HRF has ~1-2s delay, so 0s is still baseline)
    - Task execution: 0 to ~15s (finger tapping period)
    - Post-task recovery: 15 to 30s (HRF return to baseline, ~20-30s total)

    Baseline Correction:
    - Window: -5 to 0s (includes full pre-stimulus period)
    - Rationale: HRF onset delayed ~2s, so 0s reflects baseline state
    - Subtracts mean baseline from entire epoch (Req. 6.6)
    - Centers signal at zero concentration change before stimulus

    Args:
        raw: MNE Raw object with HbO and HbR channels (filtered 0.01-0.5 Hz)
        event_id: Mapping of event names to codes
            Example: {'block_start': 2, 'task_start': 1}
        tmin: Start time relative to event (default -5.0s)
        tmax: End time relative to event (default 30.0s)
            - Extended to capture full HRF return to baseline
        baseline: Baseline correction window (default (-5.0, 0.0))
            - Includes 0s (safe due to neurovascular delay)

    Returns:
        MNE Epochs object with HbO and HbR channels

    Raises:
        ValueError: If no events found or invalid time windows

    Notes:
        - Same events as EEG epochs (synchronized markers)
        - Longer tmax than EEG (30s vs 20s) for HRF recovery
        - Baseline includes 0s (unlike EEG which excludes -1 to 0s)
        - Both HbO and HbR channels included in same Epochs object
        - Reject epochs with excessive motion artifacts (if not caught by TDDR)

    Example:
        >>> events, event_id = mne.events_from_annotations(raw_haemo)
        >>> epochs = create_fnirs_epochs(raw_haemo, event_id={'block_start': 2})
        >>> print(f"Created {len(epochs)} epochs, {epochs.info['nchan']} channels")
        >>> # Includes both HbO and HbR channels

    References:
        - Obrig & Villringer (2003). HRF temporal characteristics.
        - MNE Epochs: https://mne.tools/stable/generated/mne.Epochs.html

    Requirements: 6.5, 6.6
    """
    logger.info(
        f"Creating fNIRS epochs: tmin={tmin}s, tmax={tmax}s, baseline={baseline}"
    )

    # Validate time windows
    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be < tmax ({tmax})")
    if baseline[0] >= baseline[1]:
        raise ValueError(
            f"Baseline start ({baseline[0]}) must be < end ({baseline[1]})"
        )

    # Extract events from annotations
    events, event_id_from_annot = mne.events_from_annotations(raw)

    # Filter events to only include requested event types
    if event_id:
        # Map requested event names to codes
        event_codes = [event_id_from_annot[name] for name in event_id if name in event_id_from_annot]
        if not event_codes:
            raise ValueError(
                f"No matching events found. Requested: {list(event_id.keys())}, "
                f"Available: {list(event_id_from_annot.keys())}"
            )
        # Filter events array to only include requested codes
        mask = np.isin(events[:, 2], event_codes)
        events = events[mask]
        logger.info(f"Found {len(events)} events matching {list(event_id.keys())}")
    else:
        logger.info(f"Using all {len(events)} events from annotations")

    if len(events) == 0:
        raise ValueError("No events found in raw data annotations")

    # Create epochs
    # Pick only HbO and HbR channels (exclude short channels if present)
    picks = mne.pick_types(raw.info, fnirs="hbo") + mne.pick_types(raw.info, fnirs="hbr")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id if event_id else event_id_from_annot,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True,
        proj=False,
        reject_by_annotation=True,
        verbose=True,
    )

    logger.info(
        f"Created {len(epochs)} fNIRS epochs with {epochs.info['nchan']} channels "
        f"(HbO and HbR)"
    )

    return epochs


def identify_motor_roi_channel(
    raw: mne.io.Raw,
    montage_config: dict,
    target_region: str = "C3",
) -> str:
    """
    Identify fNIRS channel closest to motor cortex region (C3 for right-hand task).

    Pilot data candidates near C3:
    - "CCP3h-CP3" (channels 12-13): Likely closest to C3
    - "FCC3h-FC3" (channels 24-25): Slightly anterior
    - "CCP1h-CP1" (channels 8-9): Medial alternative

    Algorithm:
    1. Extract channel positions from montage_config or raw.info
    2. Extract C3 position from 10-20 system
    3. Compute Euclidean distance from each fNIRS channel to C3
    4. Return channel name with minimum distance

    Args:
        raw: MNE Raw object with HbO/HbR channels
        montage_config: Channel montage with 3D positions (optional, can use raw.info)
        target_region: EEG channel name for target region (default 'C3')

    Returns:
        Channel name closest to target region (e.g., 'CCP3h-CP3 hbo')

    Raises:
        ValueError: If target region not found or no valid channels

    Notes:
        - Use HbO channel for analysis (HbR shows inverse pattern)
        - Verify channel not marked as bad during quality assessment
        - Can visualize layout to confirm spatial correspondence
        - Alternative: Manual selection based on montage inspection

    Example:
        >>> motor_channel = identify_motor_roi_channel(raw_haemo, montage_config)
        >>> print(f"Motor ROI channel: {motor_channel}")
        >>> # Output: "Motor ROI channel: CCP3h-CP3 hbo"

    Requirements: 6.7
    """
    logger.info(f"Identifying fNIRS channel closest to {target_region}")

    # Get target position from standard 10-20 montage
    standard_montage = mne.channels.make_standard_montage("standard_1020")
    target_pos = None

    for ch_name, pos in zip(standard_montage.ch_names, standard_montage.get_positions()["ch_pos"].values()):
        if ch_name == target_region:
            target_pos = np.array(pos)
            break

    if target_pos is None:
        raise ValueError(
            f"Target region '{target_region}' not found in standard 10-20 montage"
        )

    logger.info(f"Target position for {target_region}: {target_pos}")

    # Get fNIRS channel positions from raw.info
    fnirs_positions = raw.get_montage().get_positions()
    ch_pos = fnirs_positions["ch_pos"]

    # Find HbO channels only (we'll use HbO for analysis)
    hbo_channels = [ch for ch in raw.ch_names if "hbo" in ch.lower()]

    if not hbo_channels:
        raise ValueError("No HbO channels found in raw data")

    # Compute distances
    distances = {}
    for ch_name in hbo_channels:
        if ch_name in ch_pos:
            ch_position = np.array(ch_pos[ch_name])
            distance = np.linalg.norm(ch_position - target_pos)
            distances[ch_name] = distance
        else:
            logger.warning(f"Channel {ch_name} has no position information, skipping")

    if not distances:
        raise ValueError("No fNIRS channels with position information found")

    # Find channel with minimum distance
    closest_channel = min(distances, key=distances.get)
    min_distance = distances[closest_channel]

    logger.info(
        f"Closest fNIRS channel to {target_region}: {closest_channel} "
        f"(distance: {min_distance*1000:.1f} mm)"
    )

    # Check if channel is marked as bad
    if closest_channel in raw.info["bads"]:
        logger.warning(
            f"Selected channel {closest_channel} is marked as BAD. "
            f"Consider using next closest channel."
        )
        # Find next closest good channel
        good_channels = {ch: dist for ch, dist in distances.items() if ch not in raw.info["bads"]}
        if good_channels:
            closest_channel = min(good_channels, key=good_channels.get)
            min_distance = good_channels[closest_channel]
            logger.info(
                f"Using next closest good channel: {closest_channel} "
                f"(distance: {min_distance*1000:.1f} mm)"
            )
        else:
            logger.error("All fNIRS channels near target region are marked as BAD")

    return closest_channel


def extract_hrf(
    epochs: mne.Epochs,
    channel: str,
    chromophore: str = "hbo",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract averaged hemodynamic response function for a channel.

    Averages across trials to obtain grand-average HRF, reducing noise and
    revealing consistent hemodynamic pattern.

    Args:
        epochs: MNE Epochs object with fNIRS data
        channel: Channel name (e.g., 'CCP3h-CP3 hbo')
        chromophore: 'hbo' or 'hbr' (default 'hbo')
            - HbO: Expected to increase during activation
            - HbR: Expected to decrease (inverse pattern)

    Returns:
        times: Time vector (seconds, e.g., -5 to 30)
        hrf: Averaged HRF across trials (μM or μmol/L)

    Raises:
        ValueError: If channel not found or chromophore invalid

    Notes:
        - Uses epochs.average(picks=channel) to compute grand average
        - Returns Evoked object data for specified channel/chromophore
        - HRF shape: Onset ~2s, plateau ~5-15s, return ~20-30s
        - For sustained tasks (15s), expect plateau rather than sharp peak

    Example:
        >>> times, hrf_hbo = extract_hrf(epochs, 'CCP3h-CP3 hbo', chromophore='hbo')
        >>> times, hrf_hbr = extract_hrf(epochs, 'CCP3h-CP3 hbr', chromophore='hbr')
        >>> # hrf_hbo should show positive deflection during task
        >>> # hrf_hbr should show negative deflection (inverse)

    Requirements: 6.8
    """
    logger.info(f"Extracting HRF for channel: {channel}, chromophore: {chromophore}")

    # Validate chromophore
    if chromophore not in ["hbo", "hbr"]:
        raise ValueError(f"chromophore must be 'hbo' or 'hbr', got '{chromophore}'")

    # Validate channel exists
    if channel not in epochs.ch_names:
        raise ValueError(
            f"Channel '{channel}' not found in epochs. "
            f"Available channels: {epochs.ch_names}"
        )

    # Verify channel matches chromophore
    if chromophore not in channel.lower():
        logger.warning(
            f"Channel name '{channel}' does not contain '{chromophore}'. "
            f"Verify correct channel selected."
        )

    # Average across trials
    evoked = epochs.average(picks=[channel])

    # Extract time vector and data
    times = evoked.times
    hrf = evoked.get_data()[0, :]  # Shape: (1, n_times) -> (n_times,)

    logger.info(
        f"Extracted HRF: {len(times)} time points, "
        f"time range: [{times[0]:.1f}, {times[-1]:.1f}]s, "
        f"mean amplitude: {np.mean(hrf):.3f} μM"
    )

    return times, hrf


def validate_hrf_temporal_dynamics(
    times: np.ndarray,
    hrf_hbo: np.ndarray,
    epochs: mne.Epochs,
    channel: str,
    onset_window: tuple[float, float] = (2.0, 3.0),
    peak_window: tuple[float, float] = (4.0, 8.0),
    plateau_window: tuple[float, float] = (5.0, 15.0),
    baseline_window: tuple[float, float] = (-5.0, 0.0),
    onset_threshold_um: float = 0.1,
) -> dict[str, Any]:
    """
    Validate HRF temporal characteristics with statistical testing.

    Checks physiological plausibility of hemodynamic response:
    1. Onset latency: HbO begins increasing within 2-3s (neurovascular delay)
    2. Time-to-peak: Peak/plateau occurs within 4-8s (for brief stimuli)
    3. Plateau amplitude: Mean HbO during task (5-15s) significantly positive
    4. Trial consistency: Correlation between individual trials and grand average

    Onset Detection Algorithm:
    - Scan HbO from 0s onward
    - Find first time point where HbO > onset_threshold_um (e.g., 0.1 μM)
    - Alternative: Find first positive peak in derivative (dHbO/dt)
    - Verify onset falls within expected window (2-3s)

    Time-to-Peak Considerations:
    - For sustained tasks (15s tapping), may see plateau rather than sharp peak
    - Peak detection: Find maximum HbO value in 0-15s window
    - If peak occurs late (>8s), may indicate plateau rather than transient response
    - Document this possibility: Plateau is valid response for sustained stimuli

    Plateau Amplitude Statistical Test:
    - For each trial: Compute mean HbO in plateau window (5-15s)
    - For each trial: Compute mean HbO in baseline window (-5 to 0s)
    - Paired t-test: H0 = no difference between plateau and baseline
    - If p < 0.05: Plateau is statistically significant (Req. 6.10)

    Trial Consistency Metric:
    - For each trial: Extract HbO time series (0-30s)
    - Compute Pearson correlation with grand-average HRF
    - Average correlations across trials
    - Interpretation: r ≈ 1 (highly consistent), r ≤ 0.5 (variable)

    Args:
        times: Time vector (seconds)
        hrf_hbo: Averaged HbO response (μM)
        epochs: MNE Epochs object (for single-trial analysis)
        channel: Channel name for single-trial extraction
        onset_window: Expected onset latency range (default 2-3s)
        peak_window: Expected time-to-peak range (default 4-8s)
        plateau_window: Task execution window (default 5-15s)
        baseline_window: Baseline reference window (default -5 to 0s)
        onset_threshold_um: Threshold for onset detection (default 0.1 μM)

    Returns:
        Dictionary with validation results:
            - 'onset_detected': Boolean, HbO rises within onset_window
            - 'onset_time_sec': Time of onset (seconds)
            - 'time_to_peak_sec': Time of maximum HbO (seconds)
            - 'peak_within_range': Boolean, peak in 4-8s window
            - 'peak_value_um': Maximum HbO concentration (μM)
            - 'plateau_amplitude': Mean HbO during plateau (μM)
            - 'plateau_significant': Boolean, t-test p < 0.05
            - 'plateau_pvalue': Exact p-value for plateau test
            - 'trial_consistency': Mean correlation across trials (0-1)
            - 'n_trials': Number of trials used

    Notes:
        - Onset detection may fail if signal is noisy or delayed (log warning)
        - Peak detection for sustained tasks: Plateau is valid, not failure
        - Statistical significance validates robust hemodynamic response
        - Trial consistency indicates reliability of HRF measurement
        - Initial dip (0-2s HbO decrease) not required (Req. 8.7)

    Example:
        >>> validation = validate_hrf_temporal_dynamics(times, hrf_hbo, epochs, channel)
        >>> print(f"Onset: {validation['onset_time_sec']:.1f}s")
        >>> print(f"Peak: {validation['time_to_peak_sec']:.1f}s")
        >>> print(f"Plateau: {validation['plateau_amplitude']:.2f} μM (p={validation['plateau_pvalue']:.4f})")
        >>> print(f"Consistency: r={validation['trial_consistency']:.2f}")

    References:
        - Obrig & Villringer (2003). HRF onset latency (Req. 6.9).
        - Huppert et al. (2009). Time-to-peak validation (Req. 6.11).
        - Pinti et al. (2018). Plateau amplitude for sustained tasks (Req. 6.10).

    Requirements: 6.9, 6.10, 6.11
    """
    logger.info("Validating HRF temporal dynamics")

    results: dict[str, Any] = {}

    # 1. Onset Detection
    # Find first time point after 0s where HbO exceeds threshold
    post_stimulus_mask = times >= 0.0
    post_stimulus_times = times[post_stimulus_mask]
    post_stimulus_hrf = hrf_hbo[post_stimulus_mask]

    onset_indices = np.where(post_stimulus_hrf > onset_threshold_um)[0]

    if len(onset_indices) > 0:
        onset_idx = onset_indices[0]
        onset_time = post_stimulus_times[onset_idx]
        onset_detected = onset_window[0] <= onset_time <= onset_window[1]
        results["onset_detected"] = onset_detected
        results["onset_time_sec"] = float(onset_time)

        if onset_detected:
            logger.info(f"HRF onset detected at {onset_time:.2f}s (within expected {onset_window}s)")
        else:
            logger.warning(
                f"HRF onset at {onset_time:.2f}s outside expected window {onset_window}s"
            )
    else:
        results["onset_detected"] = False
        results["onset_time_sec"] = None
        logger.warning(
            f"No HRF onset detected (threshold: {onset_threshold_um} μM). "
            f"Signal may be weak or noisy."
        )

    # 2. Time-to-Peak Detection
    # Find maximum HbO in 0-15s window (covers both transient and sustained responses)
    peak_search_mask = (times >= 0.0) & (times <= 15.0)
    peak_search_hrf = hrf_hbo[peak_search_mask]
    peak_search_times = times[peak_search_mask]

    if len(peak_search_hrf) > 0:
        peak_idx = np.argmax(peak_search_hrf)
        time_to_peak = peak_search_times[peak_idx]
        peak_value = peak_search_hrf[peak_idx]
        peak_within_range = peak_window[0] <= time_to_peak <= peak_window[1]

        results["time_to_peak_sec"] = float(time_to_peak)
        results["peak_value_um"] = float(peak_value)
        results["peak_within_range"] = peak_within_range

        if peak_within_range:
            logger.info(f"HRF peak at {time_to_peak:.2f}s (within expected {peak_window}s)")
        else:
            logger.info(
                f"HRF peak at {time_to_peak:.2f}s (outside typical {peak_window}s range). "
                f"May indicate sustained plateau response for 15s task."
            )
    else:
        results["time_to_peak_sec"] = None
        results["peak_value_um"] = None
        results["peak_within_range"] = False
        logger.warning("Could not detect HRF peak")

    # 3. Plateau Amplitude Statistical Test
    # Extract single-trial data for statistical testing
    epochs_data = epochs.get_data(picks=[channel])  # Shape: (n_trials, 1, n_times)
    epochs_data = epochs_data[:, 0, :]  # Shape: (n_trials, n_times)
    n_trials = epochs_data.shape[0]

    # Find time indices for plateau and baseline windows
    plateau_mask = (times >= plateau_window[0]) & (times <= plateau_window[1])
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])

    # Compute mean amplitude in each window for each trial
    plateau_amplitudes = np.mean(epochs_data[:, plateau_mask], axis=1)
    baseline_amplitudes = np.mean(epochs_data[:, baseline_mask], axis=1)

    # Paired t-test: plateau vs baseline
    t_stat, p_value = stats.ttest_rel(plateau_amplitudes, baseline_amplitudes)

    plateau_mean = np.mean(plateau_amplitudes)
    plateau_significant = p_value < 0.05

    results["plateau_amplitude"] = float(plateau_mean)
    results["plateau_significant"] = plateau_significant
    results["plateau_pvalue"] = float(p_value)

    if plateau_significant:
        logger.info(
            f"Plateau amplitude: {plateau_mean:.3f} μM (significantly different from baseline, "
            f"p={p_value:.4f})"
        )
    else:
        logger.warning(
            f"Plateau amplitude: {plateau_mean:.3f} μM (NOT significantly different from baseline, "
            f"p={p_value:.4f})"
        )

    # 4. Trial Consistency
    # Compute correlation between each trial and grand average
    correlations = []
    for trial_idx in range(n_trials):
        trial_hrf = epochs_data[trial_idx, :]
        # Compute Pearson correlation
        corr, _ = stats.pearsonr(trial_hrf, hrf_hbo)
        correlations.append(corr)

    mean_consistency = np.mean(correlations)
    results["trial_consistency"] = float(mean_consistency)
    results["n_trials"] = int(n_trials)

    if mean_consistency > 0.7:
        logger.info(f"High trial consistency: r={mean_consistency:.2f} (>0.7)")
    elif mean_consistency > 0.5:
        logger.info(f"Moderate trial consistency: r={mean_consistency:.2f}")
    else:
        logger.warning(f"Low trial consistency: r={mean_consistency:.2f} (<0.5)")

    logger.info(f"HRF validation complete: {n_trials} trials analyzed")

    return results


def compute_hrf_quality_metrics(
    epochs: mne.Epochs,
    channel: str,
    chromophore: str = "hbo",
    plateau_window: tuple[float, float] = (5.0, 15.0),
    baseline_window: tuple[float, float] = (-5.0, 0.0),
) -> dict[str, float]:
    """
    Compute HRF quality metrics across trials.

    Metrics:
    1. **Trial-to-trial consistency**: Mean correlation between trials
    2. **Signal-to-noise ratio (SNR)**: Peak amplitude / baseline std
    3. **Canonical HRF fit quality**: R² from fitting canonical HRF model (optional)

    Trial-to-Trial Consistency:
    - Compute pairwise correlations between all trial pairs
    - Average correlations (Fisher z-transform for proper averaging)
    - High consistency (r > 0.7) indicates reliable response

    Signal-to-Noise Ratio:
    - SNR = (mean_plateau - mean_baseline) / std_baseline
    - Higher SNR indicates clearer hemodynamic response
    - Typical values: SNR > 2 (good), SNR < 1 (poor)

    Canonical HRF Fit (Optional):
    - Fit double-gamma function to averaged HRF
    - Compute R² (goodness of fit)
    - High R² indicates HRF follows expected shape

    Args:
        epochs: MNE Epochs object
        channel: Channel name
        chromophore: 'hbo' or 'hbr'
        plateau_window: Task window for SNR calculation (default 5-15s)
        baseline_window: Baseline window for SNR calculation (default -5 to 0s)

    Returns:
        Dictionary with quality metrics:
            - 'consistency': Mean trial-to-trial correlation
            - 'snr': Signal-to-noise ratio
            - 'canonical_fit_r2': R² from canonical HRF fit (optional, not implemented)

    Notes:
        - Consistency and SNR are most important metrics
        - Canonical fit is optional (requires HRF model implementation)
        - Include in quality report for transparency

    Example:
        >>> quality = compute_hrf_quality_metrics(epochs, 'CCP3h-CP3 hbo')
        >>> print(f"Consistency: r={quality['consistency']:.2f}")
        >>> print(f"SNR: {quality['snr']:.1f}")

    Requirements: 6.12
    """
    logger.info(f"Computing HRF quality metrics for {channel}")

    # Extract single-trial data
    epochs_data = epochs.get_data(picks=[channel])  # Shape: (n_trials, 1, n_times)
    epochs_data = epochs_data[:, 0, :]  # Shape: (n_trials, n_times)
    n_trials = epochs_data.shape[0]
    times = epochs.times

    # 1. Trial-to-Trial Consistency
    # Compute pairwise correlations between all trials
    correlations = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            corr, _ = stats.pearsonr(epochs_data[i, :], epochs_data[j, :])
            correlations.append(corr)

    if correlations:
        # Fisher z-transform for proper averaging
        z_scores = np.arctanh(correlations)
        mean_z = np.mean(z_scores)
        mean_consistency = np.tanh(mean_z)
    else:
        mean_consistency = np.nan
        logger.warning("Only one trial available, cannot compute consistency")

    # 2. Signal-to-Noise Ratio
    # Find time indices for plateau and baseline windows
    plateau_mask = (times >= plateau_window[0]) & (times <= plateau_window[1])
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])

    # Compute mean plateau and baseline std across trials
    plateau_amplitudes = np.mean(epochs_data[:, plateau_mask], axis=1)
    baseline_amplitudes = epochs_data[:, baseline_mask]

    mean_plateau = np.mean(plateau_amplitudes)
    mean_baseline = np.mean(baseline_amplitudes)
    std_baseline = np.std(baseline_amplitudes)

    if std_baseline > 0:
        snr = (mean_plateau - mean_baseline) / std_baseline
    else:
        snr = np.nan
        logger.warning("Baseline std is zero, cannot compute SNR")

    results = {
        "consistency": float(mean_consistency) if not np.isnan(mean_consistency) else None,
        "snr": float(snr) if not np.isnan(snr) else None,
    }

    logger.info(
        f"Quality metrics: consistency={results['consistency']:.2f}, SNR={results['snr']:.1f}"
    )

    # Note: Canonical HRF fit not implemented (optional)
    # Would require fitting double-gamma function to averaged HRF

    return results


def plot_hrf_curves(
    times: np.ndarray,
    hrf_hbo: np.ndarray,
    hrf_hbr: np.ndarray,
    epochs: mne.Epochs | None = None,
    channel: str | None = None,
    individual_trials: bool = False,
    task_window: tuple[float, float] = (0.0, 15.0),
    onset_time: float | None = None,
    peak_time: float | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Generate hemodynamic response curves with shading and annotations.

    Shows grand-average HRF with variability measures and key temporal markers.

    Visual Elements:
    - **HbO curve**: Red line (expected to increase)
    - **HbR curve**: Blue line (expected to decrease, inverse pattern)
    - **Shading**: ±1 standard deviation across trials (semi-transparent)
    - **Task window**: Gray shaded region (0-15s)
    - **Onset marker**: Vertical dashed line at onset_time (if provided)
    - **Peak marker**: Circle marker at peak_time (if provided)
    - **Baseline**: Horizontal line at 0 μM
    - **Individual trials**: Semi-transparent overlays (if individual_trials=True)

    Args:
        times: Time vector (seconds, -5 to 30)
        hrf_hbo: Averaged HbO response (μM)
        hrf_hbr: Averaged HbR response (μM)
        epochs: MNE Epochs object (for std and individual trials)
        channel: Channel name (for extracting single-trial data)
        individual_trials: If True, overlay individual trial curves
        task_window: Task execution period for shading
        onset_time: Time of HbO onset for marker (optional)
        peak_time: Time of HbO peak for marker (optional)
        output_path: Optional save path for figure

    Returns:
        Matplotlib figure with HRF curves

    Figure Annotations:
    - Title: "Hemodynamic Response - Channel {channel}"
    - X-axis: Time (s)
    - Y-axis: Concentration change (μM)
    - Legend: HbO, HbR, ±1 SD, Task period
    - Caption: Key metrics (onset, peak, plateau amplitude)

    Expected Pattern:
    - HbO: Rises ~2s, plateaus 5-15s, returns ~20-30s
    - HbR: Decreases (inverse), smaller magnitude than HbO
    - Variability: Shading shows trial-to-trial consistency

    Example:
        >>> fig = plot_hrf_curves(
        ...     times, hrf_hbo, hrf_hbr,
        ...     epochs=epochs, channel='CCP3h-CP3 hbo',
        ...     individual_trials=True,
        ...     onset_time=2.3, peak_time=6.8
        ... )
        >>> fig.savefig('derivatives/figures/sub-001_hrf.png', dpi=300)

    Notes:
        - Include in validation report (Req. 8.3)
        - Annotate with key metrics in figure caption
        - Individual trials help visualize variability (optional)
        - Standard deviation shading shows consistency

    References:
        - Scholkmann et al. (2014). HRF visualization best practices.

    Requirements: 8.3, 6.12
    """
    logger.info("Plotting HRF curves")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot task window shading
    ax.axvspan(
        task_window[0],
        task_window[1],
        alpha=0.1,
        color="gray",
        label="Task period",
    )

    # Plot baseline at 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    # Plot individual trials if requested
    if individual_trials and epochs is not None and channel is not None:
        # Extract HbO channel data
        hbo_channel = channel if "hbo" in channel.lower() else channel.replace("hbr", "hbo")
        hbr_channel = channel.replace("hbo", "hbr") if "hbo" in channel.lower() else channel

        if hbo_channel in epochs.ch_names:
            hbo_data = epochs.get_data(picks=[hbo_channel])[:, 0, :]
            for trial_idx in range(hbo_data.shape[0]):
                ax.plot(
                    times,
                    hbo_data[trial_idx, :],
                    color="red",
                    alpha=0.1,
                    linewidth=0.5,
                )

        if hbr_channel in epochs.ch_names:
            hbr_data = epochs.get_data(picks=[hbr_channel])[:, 0, :]
            for trial_idx in range(hbr_data.shape[0]):
                ax.plot(
                    times,
                    hbr_data[trial_idx, :],
                    color="blue",
                    alpha=0.1,
                    linewidth=0.5,
                )

    # Compute standard deviation if epochs provided
    if epochs is not None and channel is not None:
        hbo_channel = channel if "hbo" in channel.lower() else channel.replace("hbr", "hbo")
        hbr_channel = channel.replace("hbo", "hbr") if "hbo" in channel.lower() else channel

        if hbo_channel in epochs.ch_names:
            hbo_data = epochs.get_data(picks=[hbo_channel])[:, 0, :]
            hbo_std = np.std(hbo_data, axis=0)
            ax.fill_between(
                times,
                hrf_hbo - hbo_std,
                hrf_hbo + hbo_std,
                color="red",
                alpha=0.2,
                label="HbO ±1 SD",
            )

        if hbr_channel in epochs.ch_names:
            hbr_data = epochs.get_data(picks=[hbr_channel])[:, 0, :]
            hbr_std = np.std(hbr_data, axis=0)
            ax.fill_between(
                times,
                hrf_hbr - hbr_std,
                hrf_hbr + hbr_std,
                color="blue",
                alpha=0.2,
                label="HbR ±1 SD",
            )

    # Plot grand-average HRF curves
    ax.plot(times, hrf_hbo, color="red", linewidth=2, label="HbO (grand average)")
    ax.plot(times, hrf_hbr, color="blue", linewidth=2, label="HbR (grand average)")

    # Mark onset time
    if onset_time is not None:
        ax.axvline(
            onset_time,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Onset ({onset_time:.1f}s)",
        )

    # Mark peak time
    if peak_time is not None:
        peak_idx = np.argmin(np.abs(times - peak_time))
        peak_value = hrf_hbo[peak_idx]
        ax.plot(
            peak_time,
            peak_value,
            "o",
            color="darkred",
            markersize=10,
            label=f"Peak ({peak_time:.1f}s)",
        )

    # Labels and formatting
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Concentration change (μM)", fontsize=12)
    title = f"Hemodynamic Response Function"
    if channel:
        title += f" - {channel}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"HRF plot saved to {output_path}")

    return fig


def plot_hrf_spatial_map(
    epochs: mne.Epochs,
    montage_config: dict,
    time_window: tuple[float, float] = (5.0, 15.0),
    chromophore: str = "hbo",
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Generate spatial map of HRF amplitude across all fNIRS channels.

    Shows which brain regions exhibit strongest hemodynamic response.
    Useful for validating motor cortex localization.

    Args:
        epochs: MNE Epochs object
        montage_config: Channel montage with 3D positions
        time_window: Window for amplitude calculation (default 5-15s plateau)
        chromophore: 'hbo' or 'hbr'
        output_path: Optional save path

    Returns:
        Matplotlib figure with spatial HRF map

    Visual Elements:
    - 2D projection of channel positions
    - Color-coded by mean HbO amplitude in time_window
    - Colormap: Red (high activation), Blue (low/negative)
    - Expected: Strongest activation near C3 (motor cortex)

    Notes:
        - Optional visualization (not in core requirements)
        - Validates spatial specificity of motor response
        - Useful for quality control and publication

    Requirements: 8.3, 6.12
    """
    logger.info("Plotting HRF spatial map")

    # Get channels of specified chromophore
    if chromophore == "hbo":
        channels = [ch for ch in epochs.ch_names if "hbo" in ch.lower()]
    else:
        channels = [ch for ch in epochs.ch_names if "hbr" in ch.lower()]

    if not channels:
        raise ValueError(f"No {chromophore} channels found in epochs")

    # Extract averaged data for time window
    times = epochs.times
    time_mask = (times >= time_window[0]) & (times <= time_window[1])

    amplitudes = {}
    for ch in channels:
        evoked = epochs.average(picks=[ch])
        data = evoked.get_data()[0, :]
        mean_amplitude = np.mean(data[time_mask])
        amplitudes[ch] = mean_amplitude

    # Create spatial plot using MNE's built-in topomap
    # This requires the epochs to have proper montage information
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get channel positions
    montage = epochs.get_montage()
    if montage is None:
        logger.warning("No montage found, cannot create spatial map")
        return fig

    # Create data array for topomap
    data_array = np.array([amplitudes.get(ch, 0) for ch in channels])

    # Use MNE's plot_topomap
    from mne.viz import plot_topomap

    # Get positions for selected channels
    pos = montage.get_positions()
    ch_pos = {ch: pos["ch_pos"][ch] for ch in channels if ch in pos["ch_pos"]}

    if ch_pos:
        # Convert 3D positions to 2D for topomap
        pos_array = np.array([ch_pos[ch][:2] for ch in channels if ch in ch_pos])

        im, _ = plot_topomap(
            data_array,
            pos_array,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(None, None),
            contours=6,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean HbO amplitude (μM)", fontsize=12)

        ax.set_title(
            f"HRF Spatial Map ({chromophore.upper()}, {time_window[0]}-{time_window[1]}s)",
            fontsize=14,
            fontweight="bold",
        )
    else:
        logger.warning("Could not extract channel positions for spatial map")

    plt.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spatial map saved to {output_path}")

    return fig

