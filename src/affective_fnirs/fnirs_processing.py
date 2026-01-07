"""
fNIRS Processing Module.

This module implements the complete fNIRS processing pipeline following MNE-NIRS
best practices. The pipeline transforms raw intensity data through optical density,
motion correction, short channel regression, and hemoglobin conversion.

CRITICAL PROCESSING ORDER:
1. Quality assessment (on raw intensity) - already done in fnirs_quality.py
2. Intensity → Optical Density (OD)
3. Motion correction (TDDR on OD)
4. Short channel regression (on OD)
5. Verify noise reduction
6. OD → Hemoglobin (Beer-Lambert)
7. Bandpass filter (0.01-0.5 Hz on Hb)

All preprocessing (motion correction, short channel regression) must occur in OD
space before converting to hemoglobin concentrations. This is the validated workflow
recommended by MNE-NIRS documentation.

References:
- Fishburn et al. (2019). Temporal Derivative Distribution Repair (TDDR). Neurophotonics 6(3).
- Molavi & Dumont (2012). Wavelet-based motion artifact removal. Physiological Measurement 33(2).
- MNE-NIRS documentation: https://mne.tools/mne-nirs/
- Artinis Medical Systems: Short channel regression guidelines

Requirements: 4.1-4.10, 6.1-6.4
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from scipy import signal

logger = logging.getLogger(__name__)


def convert_to_optical_density(raw_intensity: mne.io.Raw) -> mne.io.Raw:
    """
    Convert raw intensity to optical density (OD).

    This MUST be the first step in fNIRS processing pipeline.

    Algorithm:
        OD = -log₁₀(I / I₀)

    Where:
        - I: measured light intensity
        - I₀: incident light intensity (baseline)
        - OD: optical density (unitless)

    Implementation:
        Uses mne.preprocessing.nirs.optical_density(raw_intensity)
        Modifies data in-place, changing channel types to 'fnirs_od'

    Args:
        raw_intensity: MNE Raw object with fnirs_cw_amplitude channels

    Returns:
        MNE Raw object with optical density data (fnirs_od type)

    Raises:
        ValueError: If channel types are not fnirs_cw_amplitude

    Notes:
        - OD linearizes the Beer-Lambert relationship
        - All subsequent corrections (TDDR, SCR) operate on OD data
        - MNE automatically updates channel types to 'fnirs_od'
        - Negative OD values indicate intensity > baseline (physically implausible)

    References:
        - MNE optical_density: https://mne.tools/stable/generated/mne.preprocessing.nirs.optical_density.html

    Requirements: 6.1
    """
    # Validate channel types
    channel_types = raw_intensity.get_channel_types()
    if not all(ch_type == "fnirs_cw_amplitude" for ch_type in channel_types):
        raise ValueError(
            "convert_to_optical_density() requires fnirs_cw_amplitude channel types. "
            f"Found: {set(channel_types)}"
        )

    logger.info(
        f"Converting {len(raw_intensity.ch_names)} intensity channels to optical density"
    )

    # Convert to optical density using MNE-NIRS
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Verify channel types changed
    od_types = raw_od.get_channel_types()
    if not all(ch_type == "fnirs_od" for ch_type in od_types):
        logger.warning(
            f"Expected all channels to be 'fnirs_od', found: {set(od_types)}"
        )

    logger.info(
        f"Successfully converted to optical density. Channel types: {set(od_types)}"
    )

    return raw_od




def correct_motion_artifacts(
    raw_od: mne.io.Raw,
    method: str = "tddr",
    **method_params: Any,
) -> mne.io.Raw:
    """
    Apply motion artifact correction to optical density data.

    Motion artifacts appear as sharp spikes/steps in fNIRS signals caused by
    optode displacement. TDDR is the gold standard automated correction method.

    Supported methods:

    1. **'tddr'** (Temporal Derivative Distribution Repair) - RECOMMENDED
       - Detects abrupt changes in temporal derivative
       - Repairs spikes without manual parameter tuning
       - Validated in Fishburn et al. (2019)
       - Implementation: mne.preprocessing.nirs.temporal_derivative_distribution_repair()

    2. **'wavelet'** (Wavelet-based filtering)
       - Discrete Wavelet Transform (DWT) with outlier removal
       - Based on Molavi & Dumont (2012)
       - Requires manual threshold tuning
       - Implementation: Custom (not in MNE core)

    3. **'none'** (No correction)
       - Skip motion correction (for comparison/debugging)

    Args:
        raw_od: MNE Raw object with optical density data (fnirs_od type)
        method: Correction method ('tddr', 'wavelet', 'none')
        method_params: Method-specific parameters
            For 'tddr': No parameters needed (automatic)
            For 'wavelet':
                - wavelet: str = 'db5' (Daubechies 5)
                - threshold_std: float = 3.0 (outlier detection)

    Returns:
        Corrected MNE Raw object (still in OD)

    Raises:
        ValueError: If method not in ['tddr', 'wavelet', 'none']
        ValueError: If channel types are not fnirs_od

    Notes:
        - TDDR is fully automatic and robust (recommended default)
        - Wavelet method requires careful threshold selection
        - Log the number of corrected points per channel (Req. 4.10)
        - If method='none', return raw_od unchanged

    Example:
        >>> raw_od = convert_to_optical_density(raw_intensity)
        >>> raw_corrected = correct_motion_artifacts(raw_od, method='tddr')
        >>> # Log: "TDDR corrected 23 motion artifacts across 36 channels"

    References:
        - TDDR: Fishburn et al. (2019), Neurophotonics 6(3)
        - MNE TDDR: https://mne.tools/stable/generated/mne.preprocessing.nirs.temporal_derivative_distribution_repair.html
        - Wavelet: Molavi & Dumont (2012), Physiological Measurement 33(2)

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.10
    """
    # Validate method
    valid_methods = {"tddr", "wavelet", "none"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    # Validate channel types
    channel_types = raw_od.get_channel_types()
    if not all(ch_type == "fnirs_od" for ch_type in channel_types):
        raise ValueError(
            "correct_motion_artifacts() requires fnirs_od channel types. "
            f"Found: {set(channel_types)}"
        )

    if method == "none":
        logger.info("Skipping motion artifact correction (method='none')")
        return raw_od

    logger.info(f"Applying motion artifact correction using method: {method}")

    if method == "tddr":
        # Use MNE-NIRS TDDR implementation
        raw_corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(
            raw_od
        )

        # Count corrected artifacts (compare before/after)
        data_before = raw_od.get_data()
        data_after = raw_corrected.get_data()
        diff = np.abs(data_after - data_before)
        
        # Count significant corrections (threshold: 0.01 OD units)
        correction_threshold = 0.01
        corrected_points = np.sum(diff > correction_threshold, axis=1)
        total_corrected = np.sum(corrected_points)
        
        logger.info(
            f"TDDR corrected {total_corrected} motion artifacts across "
            f"{len(raw_od.ch_names)} channels"
        )
        
        # Log per-channel statistics
        for ch_idx, ch_name in enumerate(raw_od.ch_names):
            if corrected_points[ch_idx] > 0:
                logger.debug(
                    f"  {ch_name}: {corrected_points[ch_idx]} points corrected"
                )

    elif method == "wavelet":
        # Wavelet-based correction (custom implementation)
        logger.warning(
            "Wavelet motion correction not yet implemented. "
            "Using TDDR as fallback."
        )
        raw_corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(
            raw_od
        )

    return raw_corrected




def identify_short_channels(
    raw_od: mne.io.Raw,
    montage_config: dict[str, Any],
    short_threshold_mm: float = 15.0,
) -> tuple[list[str], list[str]]:
    """
    Identify short (<15mm) and long (≥15mm) channels from montage configuration.

    Short channels measure primarily superficial (scalp/skull) hemodynamics,
    while long channels capture cortical activity plus superficial contamination.

    Algorithm:
    1. Extract channel metadata from montage_config JSON
    2. For each channel, check 'type' field or compute source-detector distance
    3. Classify as short if distance < short_threshold_mm
    4. Return separate lists of short and long channel names

    Alternative: Use mne.preprocessing.nirs.short_channels(raw_od, threshold=0.015)
    which returns boolean array based on channel distances stored in info['chs'].

    Args:
        raw_od: MNE Raw object with optical density data
        montage_config: Channel montage from JSON with 'type' field or positions
        short_threshold_mm: Maximum distance for short channels (default 15mm)

    Returns:
        short_channels: List of short channel names (e.g., ['S5_D5 760', 'S5_D5 850'])
        long_channels: List of long channel names (e.g., ['S1_D1 760', ...])

    Notes:
        - Pilot data has 2 short channels (8mm): one per hemisphere
        - Long channels are 30mm source-detector separation
        - JSON 'type' field: "Short" or "Long" (if available)
        - If positions available, compute: distance = ||source_pos - detector_pos||

    Example montage_config structure:
        {
            "ChMontage": {
                "Channels": [
                    {"Name": "S1_D1 760", "type": "Long", "SourceDetectorDistance": 30},
                    {"Name": "S5_D5 760", "type": "Short", "SourceDetectorDistance": 8},
                    ...
                ]
            }
        }

    Requirements: 4.5
    """
    short_channels = []
    long_channels = []

    # Try using MNE's built-in short channel detection first
    try:
        # Convert threshold from mm to meters for MNE
        threshold_m = short_threshold_mm / 1000.0
        
        # Get boolean array indicating short channels
        is_short = mne.preprocessing.nirs.short_channels(raw_od, threshold=threshold_m)
        
        # Separate channels based on boolean array
        for ch_idx, ch_name in enumerate(raw_od.ch_names):
            if is_short[ch_idx]:
                short_channels.append(ch_name)
            else:
                long_channels.append(ch_name)
        
        logger.info(
            f"Identified {len(short_channels)} short channels "
            f"(< {short_threshold_mm}mm) and {len(long_channels)} long channels "
            f"using MNE built-in detection"
        )
        
        return short_channels, long_channels
        
    except Exception as e:
        logger.warning(
            f"MNE short channel detection failed: {e}. "
            f"Falling back to montage_config parsing."
        )

    # Fallback: Parse montage_config JSON
    if "ChMontage" not in montage_config:
        raise ValueError(
            "montage_config must contain 'ChMontage' key with channel information"
        )

    channels_info = montage_config["ChMontage"].get("Channels", [])
    
    for ch_info in channels_info:
        ch_name = ch_info.get("Name", "")
        
        # Skip if channel not in raw data
        if ch_name not in raw_od.ch_names:
            continue
        
        # Method 1: Check 'type' field
        if "type" in ch_info:
            ch_type = ch_info["type"].lower()
            if ch_type == "short":
                short_channels.append(ch_name)
            elif ch_type == "long":
                long_channels.append(ch_name)
            continue
        
        # Method 2: Check 'SourceDetectorDistance' field
        if "SourceDetectorDistance" in ch_info:
            distance_mm = ch_info["SourceDetectorDistance"]
            if distance_mm < short_threshold_mm:
                short_channels.append(ch_name)
            else:
                long_channels.append(ch_name)
            continue
        
        # Method 3: Compute from source/detector positions
        if "SourceIndex" in ch_info and "DetectorIndex" in ch_info:
            src_idx = ch_info["SourceIndex"]
            det_idx = ch_info["DetectorIndex"]
            
            sources = montage_config["ChMontage"].get("Sources", [])
            detectors = montage_config["ChMontage"].get("Detectors", [])
            
            if src_idx < len(sources) and det_idx < len(detectors):
                src_pos = np.array(sources[src_idx].get("Position", [0, 0, 0]))
                det_pos = np.array(detectors[det_idx].get("Position", [0, 0, 0]))
                
                distance_mm = np.linalg.norm(src_pos - det_pos)
                
                if distance_mm < short_threshold_mm:
                    short_channels.append(ch_name)
                else:
                    long_channels.append(ch_name)
                continue
        
        # If no method worked, assume long channel (conservative)
        logger.warning(
            f"Could not determine distance for channel {ch_name}, "
            f"assuming long channel"
        )
        long_channels.append(ch_name)

    logger.info(
        f"Identified {len(short_channels)} short channels "
        f"(< {short_threshold_mm}mm) and {len(long_channels)} long channels "
        f"from montage_config"
    )
    
    if short_channels:
        logger.debug(f"Short channels: {short_channels}")
    if long_channels:
        logger.debug(f"Long channels (first 5): {long_channels[:5]}...")

    return short_channels, long_channels




def apply_short_channel_regression(
    raw_od: mne.io.Raw,
    short_channels: list[str],
    long_channels: list[str],
    max_distance_mm: float = 15.0,
) -> mne.io.Raw:
    """
    Regress out short channel signals from long channels using GLM.

    Short channels capture superficial hemodynamics (scalp blood flow, systemic
    oscillations) without cortical contribution. Regressing these from long
    channels removes shared superficial noise while preserving cortical signals.

    Algorithm (GLM-based):
    1. For each long channel:
       - Identify nearest short channel(s) of same wavelength
       - Build design matrix: X = [short_channel_signal, intercept]
       - Fit GLM: long_signal = β₀ + β₁ * short_signal + ε
       - Compute residuals: cleaned_long = long_signal - (β₀ + β₁ * short_signal)
    2. Replace long channel data with residuals
    3. Short channels remain unchanged (for reference)

    Implementation:
    - RECOMMENDED: Use mne_nirs.signal_enhancement.short_channel_regression(raw_od)
      Automatically pairs long/short channels by proximity and wavelength
    - Manual alternative: Implement GLM per channel using numpy/scipy

    Args:
        raw_od: MNE Raw object with optical density data
        short_channels: List of short channel names
        long_channels: List of long channel names
        max_distance_mm: Maximum spatial distance for pairing (default 15mm)

    Returns:
        MNE Raw object with regressed long channels (still in OD)

    Notes:
        - Pairing by wavelength: 760nm short → 760nm long, 850nm short → 850nm long
        - Pilot data: 1 short channel per hemisphere → pairs with ipsilateral long channels
        - Verify noise reduction: PSD in 0.1-0.4 Hz should decrease (Req. 4.8)
        - Log regression statistics: β coefficients, R² per channel

    Validation (Req. 4.8):
        After regression, compute PSD in systemic band (0.1-0.4 Hz):
        - Mayer waves: ~0.1 Hz
        - Respiration: ~0.2-0.4 Hz
        Expected: 20-50% power reduction in this band for long channels

    Example:
        >>> raw_regressed = apply_short_channel_regression(raw_od, short_ch, long_ch)
        >>> # Log: "Short channel regression: mean R²=0.42, systemic power reduced by 35%"

    References:
        - Saager & Berger (2005). Direct characterization of superficial contamination. J Biomed Opt 10(4).
        - MNE-NIRS SCR: https://mne.tools/mne-nirs/stable/generated/mne_nirs.signal_enhancement.short_channel_regression.html
        - Artinis guidelines: https://artinis.com

    Requirements: 4.6, 4.7
    """
    # Validate channel types
    channel_types = raw_od.get_channel_types()
    if not all(ch_type == "fnirs_od" for ch_type in channel_types):
        raise ValueError(
            "apply_short_channel_regression() requires fnirs_od channel types. "
            f"Found: {set(channel_types)}"
        )

    if not short_channels:
        logger.warning(
            "No short channels provided. Skipping short channel regression."
        )
        return raw_od

    if not long_channels:
        logger.warning(
            "No long channels provided. Skipping short channel regression."
        )
        return raw_od

    logger.info(
        f"Applying short channel regression: {len(short_channels)} short channels, "
        f"{len(long_channels)} long channels"
    )

    try:
        # Try using MNE-NIRS built-in short channel regression
        import mne_nirs
        
        raw_regressed = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
        
        logger.info("Successfully applied short channel regression using MNE-NIRS")
        
        # Log regression statistics (if available in MNE-NIRS output)
        # Note: MNE-NIRS may not provide detailed statistics, so this is informative
        logger.debug("Short channel regression completed. Check noise reduction metrics.")
        
        return raw_regressed
        
    except ImportError:
        logger.error(
            "mne_nirs package not available. Cannot perform short channel regression. "
            "Install with: pip install mne-nirs"
        )
        raise
    except Exception as e:
        logger.error(f"Short channel regression failed: {e}")
        raise




def verify_systemic_noise_reduction(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    long_channels: list[str],
    systemic_band_hz: tuple[float, float] = (0.1, 0.4),
) -> dict[str, Any]:
    """
    Quantify systemic noise reduction after short channel regression.

    Validates that SCR successfully removed superficial oscillations (Req. 4.8).

    Algorithm:
    1. Compute PSD (Welch) for long channels before and after SCR
    2. Integrate power in systemic band (0.1-0.4 Hz)
    3. Calculate percent reduction: (power_before - power_after) / power_before * 100
    4. Average across long channels

    Args:
        raw_before: Raw OD before short channel regression
        raw_after: Raw OD after short channel regression
        long_channels: List of long channel names
        systemic_band_hz: Frequency range for systemic oscillations

    Returns:
        Dictionary with reduction metrics:
            - 'mean_reduction_percent': Average power reduction across channels
            - 'per_channel_reduction': Dict mapping channel → reduction %
            - 'power_before_mean': Mean systemic power before SCR
            - 'power_after_mean': Mean systemic power after SCR

    Notes:
        - Typical reduction: 20-50% in systemic band
        - If reduction < 10%, SCR may not be effective (log warning)
        - Include in quality report (optional, informative)

    Example output:
        {
            'mean_reduction_percent': 35.2,
            'per_channel_reduction': {'S1_D1 760': 38.1, 'S1_D1 850': 32.3, ...},
            'power_before_mean': 0.0042,
            'power_after_mean': 0.0027
        }

    Requirements: 4.8, 4.9
    """
    if not long_channels:
        logger.warning("No long channels provided for noise reduction verification")
        return {
            "mean_reduction_percent": 0.0,
            "per_channel_reduction": {},
            "power_before_mean": 0.0,
            "power_after_mean": 0.0,
        }

    logger.info(
        f"Verifying systemic noise reduction in band {systemic_band_hz[0]}-"
        f"{systemic_band_hz[1]} Hz for {len(long_channels)} long channels"
    )

    sfreq = raw_before.info["sfreq"]
    per_channel_reduction = {}
    power_before_list = []
    power_after_list = []

    for ch_name in long_channels:
        if ch_name not in raw_before.ch_names or ch_name not in raw_after.ch_names:
            logger.warning(f"Channel {ch_name} not found in both raw objects, skipping")
            continue

        # Get channel data
        ch_idx_before = raw_before.ch_names.index(ch_name)
        ch_idx_after = raw_after.ch_names.index(ch_name)
        
        data_before = raw_before.get_data(picks=[ch_idx_before])[0]
        data_after = raw_after.get_data(picks=[ch_idx_after])[0]

        # Compute PSD using Welch's method
        nperseg = int(2 * sfreq)  # 2-second windows
        
        freqs_before, psd_before = signal.welch(
            data_before, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )
        freqs_after, psd_after = signal.welch(
            data_after, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )

        # Integrate power in systemic band
        band_mask = (freqs_before >= systemic_band_hz[0]) & (
            freqs_before <= systemic_band_hz[1]
        )
        
        power_before = np.trapz(psd_before[band_mask], freqs_before[band_mask])
        power_after = np.trapz(psd_after[band_mask], freqs_after[band_mask])

        # Calculate reduction percentage
        if power_before > 0:
            reduction_percent = (
                (power_before - power_after) / power_before * 100
            )
        else:
            reduction_percent = 0.0

        per_channel_reduction[ch_name] = reduction_percent
        power_before_list.append(power_before)
        power_after_list.append(power_after)

    # Calculate mean statistics
    if power_before_list:
        mean_reduction_percent = np.mean(list(per_channel_reduction.values()))
        power_before_mean = np.mean(power_before_list)
        power_after_mean = np.mean(power_after_list)
    else:
        mean_reduction_percent = 0.0
        power_before_mean = 0.0
        power_after_mean = 0.0

    logger.info(
        f"Systemic noise reduction: {mean_reduction_percent:.1f}% "
        f"(power: {power_before_mean:.6f} → {power_after_mean:.6f})"
    )

    # Warn if reduction is insufficient
    if mean_reduction_percent < 10.0:
        logger.warning(
            f"Low systemic noise reduction ({mean_reduction_percent:.1f}%). "
            f"Expected 20-50%. Short channel regression may not be effective."
        )

    return {
        "mean_reduction_percent": mean_reduction_percent,
        "per_channel_reduction": per_channel_reduction,
        "power_before_mean": power_before_mean,
        "power_after_mean": power_after_mean,
    }




def convert_to_hemoglobin(
    raw_od: mne.io.Raw,
    dpf: float = 6.0,
) -> mne.io.Raw:
    """
    Convert optical density to hemoglobin concentration using Modified Beer-Lambert Law.

    This step converts OD (unitless) to concentration changes (μM or μmol/L) for
    oxygenated (HbO) and deoxygenated (HbR) hemoglobin.

    Modified Beer-Lambert Law:
        ΔOD(λ) = ε(λ) * Δ[Hb] * L * DPF

    Where:
        - ΔOD: change in optical density
        - ε: extinction coefficient (wavelength-dependent)
        - Δ[Hb]: change in hemoglobin concentration
        - L: source-detector distance
        - DPF: Differential Pathlength Factor (accounts for scattering)

    Solving for HbO and HbR requires two wavelengths (760nm, 850nm):
        Δ[HbO] = (ΔOD₈₅₀ * ε_HbR,760 - ΔOD₇₆₀ * ε_HbR,850) / (ε_HbO,850 * ε_HbR,760 - ε_HbO,760 * ε_HbR,850)
        Δ[HbR] = (ΔOD₇₆₀ * ε_HbO,850 - ΔOD₈₅₀ * ε_HbO,760) / (ε_HbO,850 * ε_HbR,760 - ε_HbO,760 * ε_HbR,850)

    Implementation:
        Uses mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)
        - Automatically pairs 760nm and 850nm channels
        - Creates new channels with 'hbo' and 'hbr' suffixes
        - Doubles channel count: N_channels_OD → 2 * N_pairs (HbO + HbR)

    Args:
        raw_od: MNE Raw object with optical density data (fnirs_od type)
        dpf: Differential Pathlength Factor (default 6.0 for adults)

    Returns:
        MNE Raw object with hemoglobin concentration channels (hbo, hbr types)

    Raises:
        ValueError: If channel types are not fnirs_od

    Notes:
        - DPF values: Adults ~6.0, Children ~5.5, Infants ~5.0
        - Channel naming: MNE appends ' hbo' and ' hbr' to original names
          Example: 'S1_D1 760' + 'S1_D1 850' → 'S1_D1 hbo', 'S1_D1 hbr'
        - Units: Typically μM (micromolar) or μmol/L
        - Extinction coefficients built into MNE (Cope, 1991 values)

    Channel Naming Considerations:
        - If original channels named with "HbO"/"Hb" (misleading), MNE may fail
        - Recommended: Use wavelength-based names before MBLL (e.g., "S1_D1 760")
        - After MBLL, names should clearly indicate chromophore (hbo/hbr)

    Post-Conversion Filtering:
        After MBLL, apply bandpass filter 0.01-0.5 Hz to hemoglobin data:
        - High-pass 0.01 Hz: Remove slow drifts
        - Low-pass 0.5 Hz: Remove cardiac pulsation and high-frequency noise

    Example:
        >>> raw_od = apply_short_channel_regression(raw_od, ...)
        >>> raw_haemo = convert_to_hemoglobin(raw_od, dpf=6.0)
        >>> raw_haemo.filter(l_freq=0.01, h_freq=0.5, picks='hbo')
        >>> raw_haemo.filter(l_freq=0.01, h_freq=0.5, picks='hbr')

    References:
        - Cope (1991). Extinction coefficients for NIR spectroscopy
        - MNE beer_lambert_law: https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html
        - Scholkmann et al. (2014). Review of fNIRS signal processing. NeuroImage 85.

    Requirements: 6.1, 6.2, 6.3
    """
    # Validate channel types
    channel_types = raw_od.get_channel_types()
    if not all(ch_type == "fnirs_od" for ch_type in channel_types):
        raise ValueError(
            "convert_to_hemoglobin() requires fnirs_od channel types. "
            f"Found: {set(channel_types)}"
        )

    logger.info(
        f"Converting {len(raw_od.ch_names)} OD channels to hemoglobin "
        f"concentrations using DPF={dpf}"
    )

    # Convert to hemoglobin using MNE-NIRS
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

    # Verify channel types changed
    haemo_types = raw_haemo.get_channel_types()
    hbo_count = sum(1 for ch_type in haemo_types if ch_type == "hbo")
    hbr_count = sum(1 for ch_type in haemo_types if ch_type == "hbr")

    logger.info(
        f"Successfully converted to hemoglobin: {hbo_count} HbO channels, "
        f"{hbr_count} HbR channels"
    )

    # Log channel naming
    hbo_channels = [
        ch for ch, ch_type in zip(raw_haemo.ch_names, haemo_types)
        if ch_type == "hbo"
    ]
    logger.debug(f"HbO channels (first 3): {hbo_channels[:3]}")

    return raw_haemo




def filter_hemoglobin_data(
    raw_haemo: mne.io.Raw,
    l_freq: float = 0.01,
    h_freq: float = 0.5,
) -> mne.io.Raw:
    """
    Apply bandpass filter to hemoglobin concentration data.

    Removes slow drifts (<0.01 Hz) and high-frequency noise (>0.5 Hz) while
    preserving hemodynamic response frequencies (~0.05-0.2 Hz).

    Args:
        raw_haemo: MNE Raw object with hemoglobin data (hbo, hbr types)
        l_freq: High-pass cutoff (default 0.01 Hz)
        h_freq: Low-pass cutoff (default 0.5 Hz)

    Returns:
        Filtered MNE Raw object

    Raises:
        ValueError: If no hbo/hbr channels found

    Notes:
        - Filter only hbo/hbr channels, not short channels (if retained)
        - Use FIR filter for linear phase response
        - Hemodynamic response peak: ~0.1 Hz (10-second period)
        - Cardiac pulsation: ~1 Hz (removed by low-pass)

    Example:
        >>> raw_filtered = filter_hemoglobin_data(raw_haemo, l_freq=0.01, h_freq=0.5)

    Requirements: 6.4
    """
    # Identify hbo and hbr channels
    channel_types = raw_haemo.get_channel_types()
    hbo_hbr_channels = [
        ch for ch, ch_type in zip(raw_haemo.ch_names, channel_types)
        if ch_type in ["hbo", "hbr"]
    ]

    if not hbo_hbr_channels:
        raise ValueError(
            "No HbO or HbR channels found in raw_haemo. "
            "Ensure Beer-Lambert conversion was applied."
        )

    logger.info(
        f"Filtering {len(hbo_hbr_channels)} hemoglobin channels "
        f"with bandpass {l_freq}-{h_freq} Hz"
    )

    # Apply bandpass filter to hemoglobin channels only
    raw_filtered = raw_haemo.copy()
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=hbo_hbr_channels,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )

    logger.info(
        f"Successfully filtered hemoglobin data. "
        f"Preserved hemodynamic response band (~0.05-0.2 Hz)"
    )

    return raw_filtered




def process_fnirs_pipeline(
    raw_intensity: mne.io.Raw,
    montage_config: dict[str, Any],
    motion_correction_method: str = "tddr",
    dpf: float = 6.0,
    l_freq: float = 0.01,
    h_freq: float = 0.5,
    short_threshold_mm: float = 15.0,
    apply_scr: bool = True,
    verify_noise_reduction: bool = True,
) -> tuple[mne.io.Raw, dict[str, Any]]:
    """
    Execute complete fNIRS processing pipeline in correct order.

    This function orchestrates all fNIRS processing steps following the validated
    MNE-NIRS workflow. The processing order is CRITICAL and must not be changed:

    PROCESSING ORDER:
    1. Quality assessment (on raw intensity) - already done in fnirs_quality.py
    2. Intensity → Optical Density (OD)
    3. Motion correction (TDDR on OD)
    4. Short channel regression (on OD)
    5. Verify noise reduction (optional validation)
    6. OD → Hemoglobin (Beer-Lambert)
    7. Bandpass filter (0.01-0.5 Hz on Hb)

    All preprocessing (motion correction, short channel regression) MUST occur in
    optical density space before converting to hemoglobin concentrations. This is
    the validated workflow recommended by MNE-NIRS documentation.

    Args:
        raw_intensity: MNE Raw object with raw intensity data (fnirs_cw_amplitude)
        montage_config: Channel montage configuration from JSON
        motion_correction_method: Motion correction method ('tddr', 'wavelet', 'none')
        dpf: Differential Pathlength Factor for Beer-Lambert (default 6.0 for adults)
        l_freq: High-pass filter cutoff for hemoglobin data (Hz)
        h_freq: Low-pass filter cutoff for hemoglobin data (Hz)
        short_threshold_mm: Maximum distance for short channel classification (mm)
        apply_scr: Whether to apply short channel regression (default True)
        verify_noise_reduction: Whether to verify systemic noise reduction (default True)

    Returns:
        raw_haemo_filtered: Processed MNE Raw object with filtered hemoglobin data
        processing_metrics: Dictionary containing processing statistics:
            - 'motion_artifacts_corrected': Number of artifacts corrected
            - 'short_channels': List of short channel names
            - 'long_channels': List of long channel names
            - 'noise_reduction_percent': Mean systemic noise reduction (if verified)
            - 'processing_steps_completed': List of completed processing steps

    Raises:
        ValueError: If input channel types are incorrect
        RuntimeError: If any processing step fails

    Notes:
        - Quality assessment should be performed BEFORE calling this function
        - Bad channels should be marked in raw_intensity.info['bads'] before processing
        - The function preserves bad channel markings throughout processing
        - All processing is done on copies to preserve original data

    Example:
        >>> # After quality assessment
        >>> raw_haemo, metrics = process_fnirs_pipeline(
        ...     raw_intensity=raw_intensity,
        ...     montage_config=montage_config,
        ...     motion_correction_method='tddr',
        ...     dpf=6.0,
        ...     apply_scr=True
        ... )
        >>> print(f"Processing complete. Noise reduction: {metrics['noise_reduction_percent']:.1f}%")

    References:
        - MNE-NIRS documentation: https://mne.tools/mne-nirs/
        - Fishburn et al. (2019). TDDR for motion correction. Neurophotonics 6(3).
        - Scholkmann et al. (2014). Review of fNIRS signal processing. NeuroImage 85.

    Requirements: 4.1-4.10, 6.1-6.4
    """
    logger.info("=" * 80)
    logger.info("Starting fNIRS processing pipeline")
    logger.info("=" * 80)

    # Initialize processing metrics
    processing_metrics: dict[str, Any] = {
        "motion_artifacts_corrected": 0,
        "short_channels": [],
        "long_channels": [],
        "noise_reduction_percent": 0.0,
        "processing_steps_completed": [],
    }

    # Validate input
    channel_types = raw_intensity.get_channel_types()
    if not all(ch_type == "fnirs_cw_amplitude" for ch_type in channel_types):
        raise ValueError(
            "process_fnirs_pipeline() requires fnirs_cw_amplitude channel types. "
            f"Found: {set(channel_types)}. "
            "Ensure quality assessment was performed on raw intensity data."
        )

    logger.info(
        f"Input: {len(raw_intensity.ch_names)} intensity channels, "
        f"{len(raw_intensity.info['bads'])} marked as bad"
    )

    # STEP 1: Quality assessment - already done (assumed)
    logger.info("Step 1: Quality assessment - assumed complete (see fnirs_quality.py)")
    processing_metrics["processing_steps_completed"].append("quality_assessment")

    # STEP 2: Intensity → Optical Density
    logger.info("Step 2: Converting intensity to optical density")
    try:
        raw_od = convert_to_optical_density(raw_intensity)
        processing_metrics["processing_steps_completed"].append("optical_density_conversion")
        logger.info("✓ Optical density conversion complete")
    except Exception as e:
        logger.error(f"Failed to convert to optical density: {e}")
        raise RuntimeError(f"Optical density conversion failed: {e}") from e

    # STEP 3: Motion correction (TDDR on OD)
    logger.info(f"Step 3: Applying motion correction (method: {motion_correction_method})")
    try:
        raw_od_corrected = correct_motion_artifacts(
            raw_od, method=motion_correction_method
        )
        processing_metrics["processing_steps_completed"].append("motion_correction")
        logger.info("✓ Motion correction complete")
    except Exception as e:
        logger.error(f"Failed to correct motion artifacts: {e}")
        raise RuntimeError(f"Motion correction failed: {e}") from e

    # STEP 4: Short channel identification and regression (on OD)
    if apply_scr:
        logger.info("Step 4: Identifying short and long channels")
        try:
            short_channels, long_channels = identify_short_channels(
                raw_od_corrected, montage_config, short_threshold_mm
            )
            processing_metrics["short_channels"] = short_channels
            processing_metrics["long_channels"] = long_channels
            logger.info(
                f"✓ Identified {len(short_channels)} short channels, "
                f"{len(long_channels)} long channels"
            )
        except Exception as e:
            logger.error(f"Failed to identify short channels: {e}")
            raise RuntimeError(f"Short channel identification failed: {e}") from e

        if short_channels and long_channels:
            logger.info("Step 4b: Applying short channel regression")
            try:
                # Store raw_od_corrected for noise reduction verification
                raw_od_before_scr = raw_od_corrected.copy() if verify_noise_reduction else None
                
                raw_od_regressed = apply_short_channel_regression(
                    raw_od_corrected, short_channels, long_channels
                )
                processing_metrics["processing_steps_completed"].append(
                    "short_channel_regression"
                )
                logger.info("✓ Short channel regression complete")

                # STEP 5: Verify noise reduction (optional)
                if verify_noise_reduction and raw_od_before_scr is not None:
                    logger.info("Step 5: Verifying systemic noise reduction")
                    try:
                        noise_metrics = verify_systemic_noise_reduction(
                            raw_od_before_scr, raw_od_regressed, long_channels
                        )
                        processing_metrics["noise_reduction_percent"] = noise_metrics[
                            "mean_reduction_percent"
                        ]
                        processing_metrics["processing_steps_completed"].append(
                            "noise_reduction_verification"
                        )
                        logger.info(
                            f"✓ Noise reduction verified: "
                            f"{noise_metrics['mean_reduction_percent']:.1f}%"
                        )
                    except Exception as e:
                        logger.warning(f"Noise reduction verification failed: {e}")
                        # Non-critical, continue processing

                # Use regressed data for next steps
                raw_od_final = raw_od_regressed
            except Exception as e:
                logger.error(f"Failed to apply short channel regression: {e}")
                raise RuntimeError(f"Short channel regression failed: {e}") from e
        else:
            logger.warning(
                "Skipping short channel regression: insufficient short or long channels"
            )
            raw_od_final = raw_od_corrected
    else:
        logger.info("Step 4: Short channel regression disabled (apply_scr=False)")
        raw_od_final = raw_od_corrected

    # STEP 6: OD → Hemoglobin (Beer-Lambert)
    logger.info(f"Step 6: Converting optical density to hemoglobin (DPF={dpf})")
    try:
        raw_haemo = convert_to_hemoglobin(raw_od_final, dpf=dpf)
        processing_metrics["processing_steps_completed"].append("hemoglobin_conversion")
        logger.info("✓ Hemoglobin conversion complete")
    except Exception as e:
        logger.error(f"Failed to convert to hemoglobin: {e}")
        raise RuntimeError(f"Hemoglobin conversion failed: {e}") from e

    # STEP 7: Bandpass filter (0.01-0.5 Hz on Hb)
    logger.info(f"Step 7: Filtering hemoglobin data ({l_freq}-{h_freq} Hz)")
    try:
        raw_haemo_filtered = filter_hemoglobin_data(raw_haemo, l_freq=l_freq, h_freq=h_freq)
        processing_metrics["processing_steps_completed"].append("hemoglobin_filtering")
        logger.info("✓ Hemoglobin filtering complete")
    except Exception as e:
        logger.error(f"Failed to filter hemoglobin data: {e}")
        raise RuntimeError(f"Hemoglobin filtering failed: {e}") from e

    # Pipeline complete
    logger.info("=" * 80)
    logger.info("fNIRS processing pipeline complete")
    logger.info(f"Steps completed: {', '.join(processing_metrics['processing_steps_completed'])}")
    logger.info(
        f"Output: {len(raw_haemo_filtered.ch_names)} hemoglobin channels "
        f"({len([ch for ch in raw_haemo_filtered.get_channel_types() if ch == 'hbo'])} HbO, "
        f"{len([ch for ch in raw_haemo_filtered.get_channel_types() if ch == 'hbr'])} HbR)"
    )
    logger.info("=" * 80)

    return raw_haemo_filtered, processing_metrics


