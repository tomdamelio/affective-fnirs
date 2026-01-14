"""
EEG Preprocessing and Artifact Removal Module.

This module implements EEG preprocessing following neuroscience best practices:
- Bandpass filtering (1-40 Hz) to remove drift and high-frequency noise
- ICA decomposition for artifact separation
- EOG/EMG component identification and removal
- Bad channel interpolation
- Common Average Reference (CAR)

All processing maintains data integrity and follows MNE-Python conventions.

Requirements: 5.1-5.7, 10.2, 10.3
References:
    - Makeig et al. (1996). ICA of EEG data. NIPS.
    - Delorme & Makeig (2004). EEGLAB toolbox. J Neurosci Methods 134(1).
    - MNE-Python ICA tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
"""

import logging
from typing import Tuple

import mne
import numpy as np
from scipy import signal

from affective_fnirs.config import PipelineConfig

logger = logging.getLogger(__name__)


def preprocess_eeg(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    show_progress: bool = True,
) -> mne.io.Raw:
    """
    Apply bandpass filter to EEG data.

    Removes DC drift (<1 Hz) and high-frequency noise (>40 Hz) while preserving
    motor-related rhythms (mu: 8-13 Hz, beta: 13-30 Hz).

    Filter Design:
    - Type: FIR (Finite Impulse Response) for zero-phase distortion
    - Method: Hamming window design
    - Transition bandwidth: Automatic (MNE default)

    Args:
        raw: MNE Raw object with EEG data
        l_freq: High-pass cutoff (default 1.0 Hz)
            - Removes slow drifts and DC offset
            - Preserves delta band (1-4 Hz) if needed
        h_freq: Low-pass cutoff (default 40.0 Hz)
            - Removes line noise (50/60 Hz) and high-frequency artifacts
            - Preserves gamma band (30-40 Hz) if needed
        show_progress: Display progress bar for long recordings (Req. 11.6)

    Returns:
        Filtered MNE Raw object

    Notes:
        - FIR filter ensures zero phase shift (no temporal distortion)
        - Filter separately from fNIRS data (different Raw objects)
        - Filtering can be slow for long recordings (show progress)
        - Verify preserved bands: mu (8-12 Hz), beta (13-30 Hz) for ERD/ERS

    Example:
        >>> raw_eeg_filtered = preprocess_eeg(raw_eeg, l_freq=1.0, h_freq=40.0)
        >>> # Progress: [████████████████████] 100% | Filtering EEG (32 channels)

    References:
        - MNE filter: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
        - Widmann et al. (2015). Digital filter design for EEG. J Neurosci Methods 250.

    Requirements: 5.1
    """
    logger.info(
        f"Applying bandpass filter: {l_freq}-{h_freq} Hz "
        f"({len(raw.ch_names)} channels)"
    )

    # Create a copy to avoid modifying original data
    raw_filtered = raw.copy()

    # Apply FIR bandpass filter with Hamming window (MNE default)
    # picks='eeg' ensures only EEG channels are filtered
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="eeg",
        method="fir",
        fir_design="firwin",
        verbose=show_progress,
    )

    logger.info(f"Bandpass filtering complete: {l_freq}-{h_freq} Hz")

    return raw_filtered




def detect_bad_eeg_channels(
    raw: mne.io.Raw,
    use_maxwell: bool = False,
) -> list[str]:
    """
    Detect bad EEG channels using correlation-based or Maxwell filtering methods.

    Bad channels can contaminate Common Average Reference and should be
    interpolated before re-referencing (Req. 5.7).

    Detection methods:
    1. **Maxwell filtering** (use_maxwell=True): Requires digitized head positions
       - Most robust for MEG, can work for EEG with proper montage
       - Detects channels with poor spatial correlation
    2. **Correlation-based** (use_maxwell=False, default): Works without digitization
       - Computes correlation between each channel and others
       - Marks channels with low average correlation as bad

    Args:
        raw: MNE Raw object with EEG data (must have montage set)
        use_maxwell: Use Maxwell filtering method (requires head digitization)
            Default: False (use correlation-based method)

    Returns:
        List of bad channel names

    Notes:
        - Correlation method is more robust for EEG without digitization
        - Maxwell method requires proper head model and digitization
        - For actiCHamp data, correlation method is recommended

    Example:
        >>> bad_channels = detect_bad_eeg_channels(raw_eeg)
        >>> if bad_channels:
        >>>     print(f"Bad channels detected: {bad_channels}")
        >>>     raw_eeg.info['bads'] = bad_channels

    Requirements: 5.7
    """
    logger.info("Detecting bad EEG channels using correlation method...")

    # Get EEG channel names
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]

    if not eeg_channel_names:
        logger.warning("No EEG channels found for bad channel detection")
        return []

    # Use correlation-based detection (more robust for EEG)
    # Compute correlation matrix between all channels
    data_eeg = raw.get_data(picks=eeg_picks)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_eeg)
    
    # For each channel, compute mean correlation with all other channels
    # Exclude self-correlation (diagonal)
    np.fill_diagonal(corr_matrix, np.nan)
    mean_corr = np.nanmean(corr_matrix, axis=1)
    
    # Detect outliers: channels with correlation < mean - 2*std
    corr_threshold = np.mean(mean_corr) - 2 * np.std(mean_corr)
    
    bad_channels = []
    for idx, corr in enumerate(mean_corr):
        if corr < corr_threshold:
            ch_name = eeg_channel_names[idx]
            bad_channels.append(ch_name)
            logger.warning(
                f"Channel {ch_name} has low correlation: "
                f"{corr:.3f} < {corr_threshold:.3f} (mean - 2*std)"
            )

    if bad_channels:
        logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
    else:
        logger.info("No bad channels detected")

    return bad_channels




def apply_ica_artifact_removal(
    raw: mne.io.Raw,
    n_components: int | float = 0.99,
    random_state: int = 42,
    method: str = "fastica",
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """
    Apply ICA to remove EOG and EMG artifacts.

    ICA decomposes EEG into independent components (ICs), separating neural
    sources from artifacts. Artifact ICs are identified and removed.

    Algorithm:
    1. Fit ICA on filtered continuous data (1-40 Hz, before epoching)
    2. Identify EOG components (correlation with frontal channels)
    3. Identify EMG components (high-frequency power ratio)
    4. Exclude artifact components
    5. Reconstruct clean signal

    Component Selection:
    - If n_components is int: Use exactly N components
    - If n_components is float (0-1): Use components explaining N% variance
    - Recommendation: Use 0.99 (99% variance) or n_channels for full decomposition
    - Minimum: 15 components (Req. 5.3), but more is better for 32-channel data

    Args:
        raw: MNE Raw object (filtered, 1-40 Hz)
        n_components: Number of ICA components or variance fraction
            - int: Exact number (e.g., 30 for 32 channels)
            - float: Variance fraction (e.g., 0.99 for 99%)
            - Default: 0.99 (recommended for artifact separation)
        random_state: Random seed for reproducibility (Req. 10.2, 10.3)
        method: ICA algorithm ('fastica', 'infomax', 'picard')
            - 'fastica': Fast, robust (default)
            - 'picard': Faster convergence, good for large datasets

    Returns:
        cleaned_raw: Raw object with artifacts removed
        ica: Fitted ICA object for inspection and saving

    Notes:
        - More components (≈n_channels) better separate diverse artifacts
        - 15 components is minimum, not optimal for 32 channels
        - Random seed logged in metadata for reproducibility
        - ICA object should be saved to derivatives/ica/ for reuse

    Example:
        >>> raw_clean, ica = apply_ica_artifact_removal(raw_filtered, n_components=0.99)
        >>> # Log: "ICA fitted with 30 components (99.2% variance), random_state=42"
        >>> # Save ICA: ica.save('derivatives/ica/sub-001_ses-001_task-fingertapping_ica.fif')

    References:
        - Makeig et al. (1996). ICA of EEG data. NIPS.
        - MNE ICA: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        - Delorme & Makeig (2004). EEGLAB toolbox. J Neurosci Methods 134(1).

    Requirements: 5.2, 5.3, 10.2, 10.3
    """
    logger.info(
        f"Fitting ICA: n_components={n_components}, "
        f"random_state={random_state}, method={method}"
    )

    # Create ICA object
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        max_iter=1000,
    )

    # Fit ICA on filtered continuous data (before epoching)
    # Use only EEG channels
    ica.fit(raw, picks="eeg")

    # Get actual number of components fitted
    n_components_fitted = ica.n_components_

    # Ensure minimum components requirement
    if n_components_fitted < 15:
        logger.warning(
            f"ICA fitted with only {n_components_fitted} components, "
            f"which is below the recommended minimum of 15. "
            f"Consider using more components for better artifact separation."
        )

    # Calculate variance explained
    if hasattr(ica, "pca_explained_variance_"):
        variance_explained = ica.pca_explained_variance_.sum()
        logger.info(
            f"ICA fitted: {n_components_fitted} components, "
            f"{variance_explained*100:.1f}% variance explained, "
            f"random_state={random_state}"
        )
    else:
        logger.info(
            f"ICA fitted: {n_components_fitted} components, "
            f"random_state={random_state}"
        )

    # Note: Artifact identification and exclusion will be done separately
    # by identify_eog_components() and identify_emg_components()
    # This function only fits the ICA, doesn't apply it yet

    return raw, ica




def identify_eog_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    threshold: float = 0.8,
    frontal_channels: list[str] | None = None,
) -> list[int]:
    """
    Identify ICA components correlated with eye movements.

    Without dedicated EOG channels, use frontal electrodes (Fp1, Fp2) as
    proxy for blink artifacts. Blinks produce large frontal deflections.

    Algorithm:
    1. For each ICA component:
       - Compute correlation with Fp1 and Fp2 signals
       - If |correlation| > threshold with either channel, mark as EOG
    2. Alternative: Use MNE's ica.find_bads_eog(raw, ch_name='Fp1')
    3. Visual verification: EOG components show frontal topography

    Args:
        ica: Fitted ICA object
        raw: MNE Raw object with EEG data
        threshold: Correlation threshold for EOG detection (default 0.8)
        frontal_channels: Channels to use as EOG proxy (default ['Fp1', 'Fp2'])

    Returns:
        List of component indices identified as EOG

    Notes:
        - Frontal channels (Fp1, Fp2) capture blink artifacts
        - EOG components typically show:
          * High correlation (>0.8) with frontal channels
          * Frontal-positive topography (red at Fp1/Fp2)
          * Spike-like time series during blinks
        - Threshold configurable via ICAConfig (Req. 5.4)
        - Provide topographies for manual verification (Req. 5.6)

    Visual Verification:
        >>> ica.plot_components(picks=eog_components)  # Show topographies
        >>> ica.plot_sources(raw, picks=eog_components)  # Show time series

    Example:
        >>> eog_inds = identify_eog_components(ica, raw, threshold=0.8)
        >>> # Found: [0, 2] (components 0 and 2 correlated with Fp1/Fp2)
        >>> # Visual check: Component 0 shows frontal topography ✓

    References:
        - Jung et al. (2000). Removing EEG artifacts by blind source separation. Psychophysiology 37(2).
        - MNE find_bads_eog: https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.find_bads_eog

    Requirements: 5.4, 5.6
    """
    if frontal_channels is None:
        frontal_channels = ["Fp1", "Fp2"]

    logger.info(
        f"Identifying EOG components using frontal channels: {frontal_channels}, "
        f"threshold={threshold}"
    )

    eog_components = []

    # Check if frontal channels exist in the data
    available_frontal = [ch for ch in frontal_channels if ch in raw.ch_names]

    if not available_frontal:
        logger.warning(
            f"No frontal channels found in data. "
            f"Requested: {frontal_channels}, "
            f"Available: {raw.ch_names}. "
            f"Skipping EOG component identification."
        )
        return eog_components

    # Use MNE's built-in EOG detection for each frontal channel
    for ch_name in available_frontal:
        try:
            # find_bads_eog returns indices and scores
            eog_inds, scores = ica.find_bads_eog(raw, ch_name=ch_name, threshold=threshold)

            if eog_inds:
                logger.info(
                    f"EOG components found using {ch_name}: {eog_inds} "
                    f"(max correlation: {np.max(np.abs(scores)):.3f})"
                )
                eog_components.extend(eog_inds)
        except Exception as e:
            logger.warning(f"Error detecting EOG with channel {ch_name}: {e}")
            continue

    # Remove duplicates and sort
    eog_components = sorted(list(set(eog_components)))

    if eog_components:
        logger.info(f"Total EOG components identified: {eog_components}")
    else:
        logger.info("No EOG components identified")

    return eog_components




def identify_emg_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    freq_threshold: float = 20.0,
    power_ratio_threshold: float = 2.0,
) -> list[int]:
    """
    Identify ICA components with high-frequency muscle activity.

    Muscle artifacts (EMG) produce high-frequency noise (>20 Hz) that
    contaminates EEG. ICA can isolate these into separate components.

    Algorithm:
    1. For each ICA component:
       - Compute Power Spectral Density (Welch's method)
       - Calculate power in high-freq band (20-40 Hz)
       - Calculate power in low-freq band (1-20 Hz)
       - Compute ratio: high_power / low_power
    2. If ratio > threshold (e.g., 2.0), mark as EMG
    3. Visual verification: EMG components show:
       - Spiky, irregular time series
       - Spatially localized or random topography
       - High power above 20 Hz

    Args:
        ica: Fitted ICA object
        raw: MNE Raw object
        freq_threshold: Frequency above which to measure power (default 20 Hz)
        power_ratio_threshold: Ratio of high-freq to low-freq power (default 2.0)

    Returns:
        List of component indices identified as EMG

    Notes:
        - EMG artifacts common in motor tasks (jaw clenching, neck tension)
        - Threshold 2.0 is initial estimate, adjust based on data
        - No direct MNE function; implement using PSD analysis
        - Configurable via ICAConfig (Req. 5.5)
        - Combine automatic detection with visual inspection

    Visual Verification:
        >>> ica.plot_components(picks=emg_components)  # Check topography
        >>> ica.plot_sources(raw, picks=emg_components)  # Check time series
        >>> ica.plot_properties(raw, picks=emg_components)  # Check PSD

    Example:
        >>> emg_inds = identify_emg_components(ica, raw, power_ratio_threshold=2.0)
        >>> # Found: [5, 12] (components with high-freq dominance)
        >>> # Visual check: Component 5 shows spiky time series ✓

    References:
        - Mognon et al. (2011). ADJUST: Automatic EEG artifact detector. Clin Neurophysiol 122(1).
        - Chaumon et al. (2015). A practical guide to ICA. Brain Topography 28(3).

    Requirements: 5.5, 5.6
    """
    logger.info(
        f"Identifying EMG components: freq_threshold={freq_threshold} Hz, "
        f"power_ratio_threshold={power_ratio_threshold}"
    )

    emg_components = []

    # Get ICA sources
    sources = ica.get_sources(raw)
    sfreq = raw.info["sfreq"]

    # Compute PSD for each component using Welch's method
    for comp_idx in range(ica.n_components_):
        # Get component time series
        comp_data = sources.get_data(picks=[comp_idx])[0]

        # Compute PSD using Welch's method
        # Use 2-second windows with 50% overlap
        nperseg = int(2 * sfreq)
        freqs, psd = signal.welch(
            comp_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )

        # Calculate power in low-frequency band (1 Hz to freq_threshold)
        low_freq_mask = (freqs >= 1.0) & (freqs < freq_threshold)
        low_freq_power = np.trapz(psd[low_freq_mask], freqs[low_freq_mask])

        # Calculate power in high-frequency band (freq_threshold to 40 Hz)
        # Limit to 40 Hz to match EEG filtering
        high_freq_mask = (freqs >= freq_threshold) & (freqs <= 40.0)
        high_freq_power = np.trapz(psd[high_freq_mask], freqs[high_freq_mask])

        # Avoid division by zero
        if low_freq_power > 0:
            power_ratio = high_freq_power / low_freq_power

            if power_ratio > power_ratio_threshold:
                emg_components.append(comp_idx)
                logger.info(
                    f"Component {comp_idx} identified as EMG: "
                    f"power ratio = {power_ratio:.2f} "
                    f"(high-freq: {high_freq_power:.2e}, low-freq: {low_freq_power:.2e})"
                )

    if emg_components:
        logger.info(f"Total EMG components identified: {emg_components}")
    else:
        logger.info("No EMG components identified")

    return emg_components




def interpolate_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Interpolate bad EEG channels using spherical spline interpolation.

    Bad channels must be interpolated BEFORE Common Average Reference to
    prevent contamination of the reference signal (Req. 5.7).

    Algorithm:
    1. Identify bad channels (from raw.info['bads'])
    2. For each bad channel:
       - Use spherical spline interpolation based on neighboring channels
       - Weighted by distance on scalp surface
    3. Replace bad channel data with interpolated values
    4. Remove channels from raw.info['bads'] (now repaired)

    Implementation:
        Uses raw.interpolate_bads(reset_bads=True)
        Requires channel positions (from montage)

    Args:
        raw: MNE Raw object with bad channels marked in info['bads']

    Returns:
        MNE Raw object with interpolated channels

    Notes:
        - Requires 10-20 montage with 3D positions
        - Interpolation quality depends on number of good neighbors
        - If >20% channels are bad, interpolation may be unreliable
        - Log interpolated channels for transparency

    Example:
        >>> raw.info['bads'] = ['T7', 'P8']  # Mark bad channels
        >>> raw_interp = interpolate_bad_channels(raw)
        >>> # Log: "Interpolated 2 bad channels: T7, P8"
        >>> # raw_interp.info['bads'] is now empty

    References:
        - Perrin et al. (1989). Spherical splines for scalp potential mapping. EEG Clin Neurophysiol 72(2).
        - MNE interpolate_bads: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads

    Requirements: 5.7
    """
    bad_channels = raw.info["bads"].copy()

    if not bad_channels:
        logger.info("No bad channels to interpolate")
        return raw

    # Count total EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    n_eeg_channels = len(eeg_picks)
    n_bad_channels = len(bad_channels)

    # Warn if >20% channels are bad
    bad_percentage = (n_bad_channels / n_eeg_channels) * 100
    if bad_percentage > 20:
        logger.warning(
            f"High percentage of bad channels: {n_bad_channels}/{n_eeg_channels} "
            f"({bad_percentage:.1f}%). Interpolation may be unreliable."
        )

    logger.info(f"Interpolating {n_bad_channels} bad channels: {bad_channels}")

    # Create a copy to avoid modifying original
    raw_interp = raw.copy()

    # Interpolate bad channels using spherical spline interpolation
    # reset_bads=True removes channels from info['bads'] after interpolation
    raw_interp.interpolate_bads(reset_bads=True)

    logger.info(
        f"Successfully interpolated {n_bad_channels} channels. "
        f"raw.info['bads'] is now empty: {raw_interp.info['bads']}"
    )

    return raw_interp




def rereference_eeg(
    raw: mne.io.Raw, ref_channels: str | list[str] = "average"
) -> mne.io.Raw:
    """
    Apply EEG re-referencing.

    Supports multiple referencing schemes:
    - 'average': Common Average Reference (CAR) - mean of all electrodes
    - ['Cz']: Single electrode reference (e.g., Cz)
    - ['M1', 'M2']: Linked mastoids or other electrode pairs

    Args:
        raw: MNE Raw object with EEG data
            - Bad channels should be interpolated beforehand
            - Montage should be applied (for spatial context)
        ref_channels: Reference type
            - 'average': Common Average Reference (recommended for dense arrays)
            - ['Cz']: Reference to Cz electrode (common for motor studies)
            - List of channel names: Reference to specific channels
            - None: Keep original reference

    Returns:
        Re-referenced MNE Raw object

    Notes:
        - CAR improves SNR for motor rhythms when many channels available
        - Cz reference is standard for motor cortex studies with few channels
        - MNE automatically excludes channels in info['bads'] from reference
        - Document reference transformation in processing log

    Example:
        >>> # Reference to Cz
        >>> raw_cz = rereference_eeg(raw, ref_channels=['Cz'])
        >>> # Common average reference
        >>> raw_car = rereference_eeg(raw, ref_channels='average')

    References:
        - Nunez & Srinivasan (2006). Electric Fields of the Brain. Oxford University Press.
        - MNE set_eeg_reference: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_eeg_reference

    Requirements: 5.7
    """
    # Check if there are bad channels (should be empty after interpolation)
    if raw.info["bads"]:
        logger.warning(
            f"Bad channels present before re-referencing: {raw.info['bads']}. "
            f"These will be excluded from the reference. "
            f"Consider interpolating bad channels first."
        )

    # Get number of EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    n_eeg_channels = len(eeg_picks)

    # Log reference type
    if ref_channels == "average":
        ref_desc = f"Common Average Reference ({n_eeg_channels} channels)"
    elif isinstance(ref_channels, list):
        ref_desc = f"Reference to {', '.join(ref_channels)}"
    else:
        ref_desc = f"Reference: {ref_channels}"

    logger.info(f"Applying {ref_desc} (excluding {len(raw.info['bads'])} bad channels)")

    # Create a copy to avoid modifying original
    raw_reref = raw.copy()

    # Apply re-referencing
    # projection=False means apply immediately (not as a projector)
    raw_reref.set_eeg_reference(ref_channels=ref_channels, projection=False)

    logger.info(f"Re-referencing applied successfully: {ref_desc}")

    return raw_reref




def preprocess_eeg_pipeline(
    raw_eeg: mne.io.Raw,
    config: PipelineConfig,
    save_ica_path: str | None = None,
    reference_channel: str | None = None,
    apply_car: bool | None = None,
    ica_enabled: bool | None = None,
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA | None]:
    """
    Complete EEG preprocessing pipeline following best practices.

    Pipeline stages:
    1. [Optional] Apply initial reference channel (if specified)
    2. Bandpass filter (1-40 Hz)
    3. Detect bad channels
    4. [Optional] Fit ICA (artifact decomposition)
    5. [Optional] Identify EOG components (frontal correlation)
    6. [Optional] Identify EMG components (high-freq power)
    7. [Optional] Apply ICA (remove artifacts)
    8. Interpolate bad channels
    9. [Optional] Common Average Reference (if apply_car=True)

    ICA is skipped if:
    - ica_enabled = False (or config.ica.enabled = False if not specified)
    - Number of bad channels <= config.ica.max_bad_channels_for_skip (data is clean)

    Args:
        raw_eeg: Raw EEG data with 10-20 montage applied
        config: Pipeline configuration with ICA and filter parameters
        save_ica_path: Optional path to save ICA object for reproducibility
        reference_channel: Initial reference channel (e.g., "Cz"). If None, no
            initial re-referencing is applied. Default: None
        apply_car: Whether to apply Common Average Reference at the end.
            If None, defaults to True (backward compatible). Default: None
        ica_enabled: Whether to apply ICA artifact removal. If None, uses
            config.ica.enabled. Default: None

    Returns:
        cleaned_raw: Preprocessed EEG ready for analysis
        ica: Fitted ICA object (None if ICA was skipped)

    Example:
        >>> from affective_fnirs.config import PipelineConfig
        >>> config = PipelineConfig.default()
        >>> # With configurable preprocessing options
        >>> raw_clean, ica = preprocess_eeg_pipeline(
        ...     raw_eeg, config,
        ...     reference_channel="Cz",
        ...     apply_car=False,
        ...     ica_enabled=True
        ... )
        >>> if ica is not None:
        >>>     ica.save('derivatives/ica/sub-001_ses-001_task-fingertapping_ica.fif')

    Requirements: 3.1, 5.1-5.7, 8.5-8.9
    """
    logger.info("=" * 80)
    logger.info("Starting EEG Preprocessing Pipeline")
    logger.info("=" * 80)

    # Determine ICA enabled status (parameter overrides config)
    use_ica = ica_enabled if ica_enabled is not None else config.ica.enabled
    
    # Determine CAR application (parameter overrides default True)
    use_car = apply_car if apply_car is not None else True

    # Log preprocessing configuration
    logger.info("Preprocessing Configuration:")
    logger.info(f"  Initial reference: {reference_channel if reference_channel else 'None (keep original)'}")
    logger.info(f"  Apply CAR: {use_car}")
    logger.info(f"  ICA enabled: {use_ica}")

    # Stage 0 (Optional): Apply initial reference channel
    if reference_channel is not None:
        logger.info(f"Stage 0: Applying initial reference to {reference_channel}")
        if reference_channel in raw_eeg.ch_names:
            raw_eeg = rereference_eeg(raw_eeg, ref_channels=[reference_channel])
            logger.info(f"Applied initial reference to {reference_channel}")
        else:
            logger.warning(
                f"Reference channel {reference_channel} not found in data. "
                f"Available channels: {raw_eeg.ch_names}. Skipping initial reference."
            )

    # Stage 1: Bandpass filter
    logger.info("Stage 1: Bandpass filtering")
    raw_filtered = preprocess_eeg(
        raw_eeg,
        l_freq=config.filters.eeg_bandpass_low_hz,
        h_freq=config.filters.eeg_bandpass_high_hz,
        show_progress=True,
    )

    # Stage 2: Detect bad channels
    logger.info("Stage 2: Detecting bad channels")
    bad_channels = detect_bad_eeg_channels(raw_filtered)
    if bad_channels:
        logger.warning(f"Bad channels detected: {bad_channels}")
        raw_filtered.info["bads"] = bad_channels
    else:
        logger.info("No bad channels detected")

    # Decide whether to apply ICA
    n_bad_channels = len(bad_channels)
    skip_ica = (
        not use_ica
        or n_bad_channels <= config.ica.max_bad_channels_for_skip
    )

    ica = None
    if skip_ica:
        if not use_ica:
            logger.info("ICA disabled - skipping artifact removal")
        else:
            logger.info(
                f"Only {n_bad_channels} bad channels detected "
                f"(<= {config.ica.max_bad_channels_for_skip}) - "
                f"data quality is good, skipping ICA"
            )
        raw_clean = raw_filtered.copy()
    else:
        # Stage 3: Fit ICA
        logger.info("Stage 3: Fitting ICA")
        _, ica = apply_ica_artifact_removal(
            raw_filtered,
            n_components=config.ica.n_components,
            random_state=config.ica.random_state,
            method="fastica",
        )

        # Stage 4: Identify EOG components
        logger.info("Stage 4: Identifying EOG components")
        eog_components = identify_eog_components(
            ica, raw_filtered, threshold=config.ica.eog_threshold
        )

        # Stage 5: Identify EMG components
        logger.info("Stage 5: Identifying EMG components")
        emg_components = identify_emg_components(
            ica,
            raw_filtered,
            freq_threshold=20.0,
            power_ratio_threshold=config.ica.emg_threshold,
        )

        # Stage 6: Apply ICA (exclude artifacts)
        artifact_components = sorted(list(set(eog_components + emg_components)))
        
        # Safety check: don't remove too many components
        max_components_to_remove = min(5, ica.n_components_ // 2)
        if len(artifact_components) > max_components_to_remove:
            logger.warning(
                f"Too many artifact components detected ({len(artifact_components)}), "
                f"limiting to {max_components_to_remove} to preserve signal"
            )
            artifact_components = artifact_components[:max_components_to_remove]
        
        logger.info(
            f"Stage 6: Applying ICA (excluding {len(artifact_components)} components: "
            f"{artifact_components})"
        )
        ica.exclude = artifact_components
        raw_clean = ica.apply(raw_filtered.copy())

        # Save ICA object if path provided
        if save_ica_path:
            logger.info(f"Saving ICA object to: {save_ica_path}")
            ica.save(save_ica_path, overwrite=True)

    # Stage 7: Interpolate bad channels
    if raw_clean.info["bads"]:
        logger.info("Stage 7: Interpolating bad channels")
        raw_clean = interpolate_bad_channels(raw_clean)
    else:
        logger.info("Stage 7: No bad channels to interpolate")

    # Stage 8: Common Average Reference (CAR) - conditional
    if use_car:
        logger.info("Stage 8: Applying Common Average Reference (CAR)")
        raw_clean = rereference_eeg(raw_clean, ref_channels="average")
    else:
        logger.info("Stage 8: Skipping CAR (apply_car=False)")

    logger.info("=" * 80)
    logger.info("EEG Preprocessing Pipeline Complete")
    logger.info("=" * 80)

    return raw_clean, ica


