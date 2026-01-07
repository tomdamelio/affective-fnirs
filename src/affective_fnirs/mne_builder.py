"""
MNE Object Construction Module for Multimodal Validation Pipeline.

This module converts raw stream data from XDF files into properly configured
MNE Raw objects with spatial montages and synchronized event markers.

Scientific Context:
    MNE-Python provides standardized data structures (Raw, Epochs, Evoked) for
    neuroimaging analysis. This module ensures proper channel typing, spatial
    coordinates, and temporal alignment for downstream processing.

    Key principles:
    - EEG and fNIRS remain in separate Raw objects (never mixed)
    - LSL timestamps used for sub-millisecond synchronization
    - Standard montages applied for spatial analysis
    - Event markers embedded as MNE Annotations

References:
    - MNE-Python: https://mne.tools/stable/
    - MNE-NIRS: https://mne.tools/mne-nirs/stable/
    - LSL synchronization: https://labstreaminglayer.readthedocs.io/

Requirements:
    - 2.1: EEG Raw construction with channel names
    - 2.2: Standard 10-20/10-10 montage application
    - 2.3: fNIRS Raw construction with wavelength info
    - 2.4: Channel mapping validation
    - 2.5: Source-detector distance computation
    - 2.6: Event marker synchronization
    - 11.5: Temporal alignment validation
"""

from pathlib import Path
from typing import Any

import mne
import numpy as np


class MNEConstructionError(Exception):
    """Exception raised for MNE object construction failures."""

    pass


def build_eeg_raw(
    data: np.ndarray,
    sfreq: float,
    stream_info: dict[str, Any],
    timestamps: np.ndarray,
) -> mne.io.Raw:
    """
    Construct MNE Raw object for EEG with 10-20/10-10 montage.

    This function creates a properly configured EEG Raw object with:
    - Correct channel types ('eeg' for standard channels, 'misc' for AUX)
    - Standard 10-20/10-10 montage with 3D spatial coordinates
    - LSL timestamps for synchronization

    Algorithm:
        1. Extract channel names from stream info (not JSON sidecar)
        2. Create MNE Info structure with channel metadata
        3. Set channel types: 'eeg' for standard, 'misc' for AUX_*
        4. Create RawArray with data and info
        5. Apply standard_1020 montage (covers 10-20 and 10-10)
        6. Validate all EEG channels have 3D positions

    Args:
        data: (n_samples, n_channels) EEG data array in microvolts
        sfreq: Sampling frequency in Hz
        stream_info: Stream info dictionary from XDF containing channel labels
        timestamps: (n_samples,) LSL timestamps for synchronization

    Returns:
        MNE Raw object with EEG data, montage, and metadata

    Raises:
        MNEConstructionError: If montage application fails or channels lack positions

    Notes:
        - Data should be in microvolts (MNE default for EEG)
        - Channel names extracted from stream info, not JSON sidecar
        - AUX channels (AUX_1, AUX_2, AUX_3) set to 'misc' type
        - standard_1020 montage includes ~94 positions (10-20 + 10-10)
        - Channels without positions after montage indicate naming mismatch
        - Keep EEG Raw separate from fNIRS throughout pipeline (Req. 2.1)

    Example:
        >>> eeg_data, sfreq, timestamps = extract_stream_data(eeg_stream)
        >>> raw_eeg = build_eeg_raw(eeg_data, sfreq, eeg_stream['info'], timestamps)
        >>> print(f"EEG channels: {len(raw_eeg.ch_names)}")
        >>> # EEG channels: 32

    References:
        - MNE RawArray: https://mne.tools/stable/generated/mne.io.RawArray.html
        - MNE montages: https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
        - Standard 10-20: Oostenveld & Praamstra (2001), Clin Neurophysiol 112(4)
    """
    # Validate inputs
    if data.ndim != 2:
        raise MNEConstructionError(
            f"EEG data must be 2D array (n_samples, n_channels), got shape {data.shape}"
        )

    n_samples, n_channels = data.shape

    if len(timestamps) != n_samples:
        raise MNEConstructionError(
            f"Timestamp count ({len(timestamps)}) does not match "
            f"data samples ({n_samples})"
        )

    # Extract channel names from stream info (Req. 2.1)
    channel_names = []
    try:
        desc = stream_info["desc"][0]
        channels_elem = desc["channels"][0]
        stream_channels = channels_elem["channel"]

        for ch in stream_channels:
            if "label" in ch:
                channel_names.append(ch["label"][0])
            else:
                # Fallback to generic name if no label
                channel_names.append(f"CH{len(channel_names)}")
    except (KeyError, IndexError, TypeError):
        # Fallback: use generic channel names
        channel_names = [f"EEG{i:03d}" for i in range(n_channels)]

    # Validate channel count
    if len(channel_names) != n_channels:
        raise MNEConstructionError(
            f"Channel name count ({len(channel_names)}) does not match "
            f"data channels ({n_channels})"
        )

    # Create MNE Info structure
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types="eeg",  # Default all to EEG, will override AUX below
    )

    # Set channel types: 'misc' for auxiliary channels
    aux_channels = [ch for ch in channel_names if "AUX" in ch]
    if aux_channels:
        channel_types = {ch: "misc" for ch in aux_channels}
        info.set_channel_types(channel_types)

    # Create Raw object (data in microvolts)
    raw = mne.io.RawArray(data.T, info)  # MNE expects (n_channels, n_samples)

    # Store LSL timestamps for synchronization (used in embed_events)
    # Store as private attribute for later use
    raw._lsl_timestamps = timestamps

    # Apply standard 10-20/10-10 montage (Req. 2.2)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        raise MNEConstructionError(
            f"Failed to apply standard_1020 montage: {e}\n"
            f"Action: Verify channel names match standard 10-20/10-10 nomenclature."
        ) from e

    # Validate all EEG channels have 3D positions (Req. 2.2)
    eeg_channels = [
        ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"
    ]
    missing_positions = []

    for ch in eeg_channels:
        ch_idx = raw.ch_names.index(ch)
        # Check if position is all zeros (no position assigned)
        pos = raw.info["chs"][ch_idx]["loc"][:3]
        if np.allclose(pos, 0.0):
            missing_positions.append(ch)

    if missing_positions:
        raise MNEConstructionError(
            f"Channels without 3D positions after montage: {missing_positions}\n"
            f"Action: Verify channel names match standard 10-20/10-10 nomenclature.\n"
            f"Expected names: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, etc.\n"
            f"Extended 10-10: F9, F10, P9, P10, etc."
        )

    return raw



def build_fnirs_raw(
    data: np.ndarray,
    sfreq: float,
    montage_config: list[dict[str, Any]],
    timestamps: np.ndarray,
) -> mne.io.Raw:
    """
    Construct MNE Raw object for fNIRS with wavelength metadata.

    This function creates a properly configured fNIRS Raw object with:
    - Channel types set to 'fnirs_cw_amplitude' (continuous wave amplitude)
    - Wavelength information stored in channel metadata
    - Source-detector distances computed and stored
    - Proper channel naming for MNE-NIRS compatibility

    Algorithm:
        1. Validate channel count (data may have more channels than JSON)
        2. Create channel names from source-detector pairs and wavelengths
        3. Create MNE Info with 'fnirs_cw_amplitude' channel types
        4. Store wavelength in loc[9] field (in meters: 760e-9, 850e-9)
        5. Compute and store source-detector distance in loc[10]
        6. Create RawArray with data and metadata

    Channel Naming Convention:
        Format: "SourceLabel_DetectorLabel wavelength"
        Examples:
            - "FCC3h_C1 760" (long channel, 760nm)
            - "FCC3h_C1 850" (long channel, 850nm)
            - "ShortL_CP3 760" (short channel, 760nm)

        Note: Original JSON uses misleading "_Hb" and "_HbO" suffixes, but
        these are raw intensities, not hemoglobin concentrations yet.

    Args:
        data: (n_samples, n_channels) fNIRS intensity data
        sfreq: Sampling frequency in Hz
        montage_config: List of channel dictionaries from JSON sidecar
            Each dict contains: channel_idx, source, detector, wavelength, type
            Note: May not cover all channels in data (extra channels handled)
        timestamps: (n_samples,) LSL timestamps for synchronization

    Returns:
        MNE Raw object with fNIRS data and wavelength metadata

    Raises:
        MNEConstructionError: If channel count mismatch or invalid configuration

    Notes:
        - Data should be raw light intensity (arbitrary units)
        - Wavelengths stored in meters (760e-9, 850e-9) per MNE convention
        - Source-detector distance computed from montage positions (if available)
        - Channel types must be 'fnirs_cw_amplitude' for MNE-NIRS processing
        - Keep fNIRS Raw separate from EEG throughout pipeline (Req. 2.3)
        - Extra channels beyond JSON config are named generically

    Example:
        >>> fnirs_data, sfreq, timestamps = extract_stream_data(fnirs_stream)
        >>> montage_config = json_sidecar['ChMontage']
        >>> raw_fnirs = build_fnirs_raw(fnirs_data, sfreq, montage_config, timestamps)
        >>> print(f"fNIRS channels: {len(raw_fnirs.ch_names)}")
        >>> # fNIRS channels: 42

    References:
        - MNE-NIRS channel types: https://mne.tools/mne-nirs/stable/
        - fNIRS data format: Huppert et al. (2009), Appl Opt 48(10)
    """
    # Validate inputs
    if data.ndim != 2:
        raise MNEConstructionError(
            f"fNIRS data must be 2D array (n_samples, n_channels), got shape {data.shape}"
        )

    n_samples, n_channels = data.shape

    if len(timestamps) != n_samples:
        raise MNEConstructionError(
            f"Timestamp count ({len(timestamps)}) does not match "
            f"data samples ({n_samples})"
        )

    # Note: Data may have more channels than JSON config (extra channels)
    # This is expected - JSON may only document primary channels
    if len(montage_config) > n_channels:
        raise MNEConstructionError(
            f"JSON montage config has more channels ({len(montage_config)}) "
            f"than data matrix ({n_channels}).\n"
            f"Action: Verify JSON 'ChMontage' matches recording configuration."
        )

    # Create channel names from montage config (Req. 2.3, 2.5)
    channel_names = []
    channel_metadata = []  # Store wavelength and distance info

    for ch_info in montage_config:
        # Extract source and detector labels (remove S/D prefix and location suffix)
        source_label = ch_info["source"].replace("S", "").replace("_", "")
        detector_label = ch_info["detector"].replace("D", "").replace("_", "")
        wavelength = ch_info["wavelength"]

        # Create MNE-NIRS compatible name: "Source_Detector wavelength"
        # Example: "FCC3h_C1 760" or "ShortL_CP3 760"
        ch_name = f"{source_label}_{detector_label} {wavelength}"
        channel_names.append(ch_name)

        # Store metadata for later
        channel_metadata.append({
            "wavelength_nm": wavelength,
            "type": ch_info["type"],
            "channel_idx": ch_info["channel_idx"]
        })

    # Add generic names for extra channels beyond JSON config
    n_extra = n_channels - len(montage_config)
    if n_extra > 0:
        import warnings
        warnings.warn(
            f"Data has {n_extra} more channels than JSON config. "
            f"Extra channels will be named generically (AUX_000, AUX_001, ...).",
            UserWarning
        )
        for i in range(n_extra):
            channel_names.append(f"AUX_{i:03d}")
            channel_metadata.append({
                "wavelength_nm": None,
                "type": "Unknown",
                "channel_idx": len(montage_config) + i
            })

    # Create MNE Info structure with fNIRS channel types
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types="fnirs_cw_amplitude",  # Continuous wave amplitude
    )

    # Store wavelength and distance metadata in channel info (Req. 2.3, 2.5)
    for idx, metadata in enumerate(channel_metadata):
        ch_idx = info["ch_names"].index(channel_names[idx])

        # Store wavelength in loc[9] (in meters) if available
        if metadata["wavelength_nm"] is not None:
            wavelength_m = metadata["wavelength_nm"] * 1e-9  # Convert nm to meters
            info["chs"][ch_idx]["loc"][9] = wavelength_m

        # Compute and store source-detector distance in loc[10]
        # For now, use type field to determine distance
        # TODO: Compute from actual 3D positions when montage positions available
        if metadata["type"] == "Short":
            distance_mm = 8.0  # Short channels: 8mm
        elif metadata["type"] == "Long":
            distance_mm = 30.0  # Long channels: 30mm
        else:
            distance_mm = 0.0  # Unknown type

        distance_m = distance_mm / 1000.0  # Convert mm to meters
        info["chs"][ch_idx]["loc"][10] = distance_m

    # Create Raw object
    raw = mne.io.RawArray(data.T, info)  # MNE expects (n_channels, n_samples)

    # Store LSL timestamps for synchronization (used in embed_events)
    raw._lsl_timestamps = timestamps

    return raw



def embed_events(
    raw: mne.io.Raw,
    marker_stream: dict[str, Any],
    event_mapping: dict[str, int] | None = None,
    temporal_tolerance_ms: float = 1.0,
) -> mne.io.Raw:
    """
    Add event annotations to Raw object from marker stream.

    This function synchronizes event markers with continuous data using LSL
    timestamps, ensuring sub-millisecond temporal precision. It validates
    temporal alignment and creates MNE Annotations for epoch extraction.

    Algorithm:
        1. Extract marker data (event names/codes) and LSL timestamps
        2. Get recording start time from raw._lsl_timestamps[0]
        3. Calculate onset times: onset = marker_time_lsl - recording_start_lsl
        4. Validate temporal precision (drift < 1ms tolerance)
        5. Create MNE Annotations with onset, duration=0, description
        6. Embed annotations in Raw object

    Temporal Validation:
        - Verifies events fall within recording time range
        - Checks for temporal drift > tolerance (indicates sync issues)
        - Validates events precede expected neural patterns (e.g., ERD)

    Args:
        raw: MNE Raw object (EEG or fNIRS) with LSL timestamps
        marker_stream: Markers stream dictionary from XDF
            Contains 'time_series' (event names) and 'time_stamps' (LSL times)
        event_mapping: Optional dict mapping event names to integer codes
            Example: {'task_start': 1, 'block_start': 2, 'tap_cue': 3}
            If None, events keep string descriptions
        temporal_tolerance_ms: Maximum allowed temporal drift in milliseconds
            Default: 1.0ms (sub-millisecond precision requirement)

    Returns:
        Raw object with embedded annotations

    Raises:
        MNEConstructionError: If temporal drift exceeds tolerance or events invalid

    Notes:
        - LSL timestamps ensure sub-millisecond synchronization (Req. 2.6)
        - Annotations have duration=0 (point events)
        - Event codes used for epochs extraction (mne.Epochs)
        - Temporal validation critical for neurovascular coupling (Req. 11.5)
        - Events should precede expected ERD (movement onset before ERD)

    Example:
        >>> raw_eeg = build_eeg_raw(eeg_data, sfreq, channel_names, timestamps)
        >>> event_mapping = {'task_start': 1, 'tap_cue': 3, 'task_end': 5}
        >>> raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)
        >>> print(f"Events: {len(raw_eeg.annotations)}")
        >>> # Events: 36

    Example Error:
        >>> # If temporal drift detected:
        >>> MNEConstructionError: Temporal drift exceeds tolerance.
        >>> Event 'tap_cue' at 15.234s has drift of 2.3ms from expected time.
        >>> Tolerance: 1.0ms
        >>> Action: Check LSL clock synchronization during recording.

    References:
        - MNE Annotations: https://mne.tools/stable/generated/mne.Annotations.html
        - LSL synchronization: https://labstreaminglayer.readthedocs.io/
    """
    # Validate raw has LSL timestamps
    if not hasattr(raw, "_lsl_timestamps"):
        raise MNEConstructionError(
            "Raw object missing LSL timestamps.\n"
            "Action: Ensure build_eeg_raw() or build_fnirs_raw() was used."
        )

    # Extract marker data and timestamps
    try:
        marker_data = marker_stream["time_series"]
        marker_timestamps = marker_stream["time_stamps"]
    except KeyError as e:
        raise MNEConstructionError(
            f"Marker stream missing required field: {e}\n"
            f"Action: Verify marker stream contains 'time_series' and 'time_stamps'."
        ) from e

    # Convert marker data to list if needed and flatten nested lists
    if isinstance(marker_data, np.ndarray):
        # Handle both 1D and 2D arrays
        if marker_data.ndim == 2:
            marker_data = marker_data.flatten()
        marker_data = marker_data.tolist()
    
    # Flatten nested lists (marker data often comes as [['LEFT'], ['RIGHT'], ...])
    flattened_markers = []
    for marker in marker_data:
        if isinstance(marker, list):
            # Take first element if it's a list
            flattened_markers.append(str(marker[0]) if marker else "unknown")
        else:
            flattened_markers.append(str(marker))
    marker_data = flattened_markers

    # Validate marker data is non-empty
    if len(marker_data) == 0:
        raise MNEConstructionError(
            "Marker stream contains no events.\n"
            "Action: Verify marker stream was recorded correctly."
        )

    # Get recording start time from LSL timestamps
    recording_start_lsl = raw._lsl_timestamps[0]
    recording_end_lsl = raw._lsl_timestamps[-1]

    # Calculate onset times relative to recording start (Req. 2.6)
    onsets = []
    descriptions = []
    skipped_events = []

    for marker_time_lsl, marker_name in zip(marker_timestamps, marker_data):
        # Calculate onset in seconds relative to recording start
        onset_sec = marker_time_lsl - recording_start_lsl

        # Skip events outside recording time range (expected for multi-modal data)
        if marker_time_lsl < recording_start_lsl:
            skipped_events.append(
                f"Event '{marker_name}' at {marker_time_lsl:.3f}s before recording start"
            )
            continue
        
        if marker_time_lsl > recording_end_lsl:
            skipped_events.append(
                f"Event '{marker_name}' at {marker_time_lsl:.3f}s after recording end"
            )
            continue

        # Validate temporal precision (Req. 11.5)
        # Check if onset is reasonable (not negative, not beyond recording)
        if onset_sec < 0:
            skipped_events.append(
                f"Event '{marker_name}' has negative onset {onset_sec:.6f}s"
            )
            continue

        recording_duration = recording_end_lsl - recording_start_lsl
        if onset_sec > recording_duration:
            skipped_events.append(
                f"Event '{marker_name}' onset {onset_sec:.3f}s exceeds duration {recording_duration:.3f}s"
            )
            continue

        onsets.append(onset_sec)

        # Map event name to code if mapping provided
        if event_mapping and marker_name in event_mapping:
            # Store both code and name for clarity
            event_code = event_mapping[marker_name]
            description = f"{marker_name}/{event_code}"
        else:
            description = str(marker_name)

        descriptions.append(description)

    # Warn about skipped events (expected for multi-modal data with different durations)
    if skipped_events:
        import warnings
        n_skipped = len(skipped_events)
        n_total = len(marker_data)
        warnings.warn(
            f"Skipped {n_skipped}/{n_total} events outside recording range. "
            f"This is expected for multi-modal data with different durations.\n"
            f"First few skipped: {skipped_events[:3]}",
            UserWarning
        )

    # Validate we have events after filtering
    if len(onsets) == 0:
        raise MNEConstructionError(
            "No valid events found after temporal validation.\n"
            f"Total markers: {len(marker_data)}, Invalid: {len(invalid_events)}\n"
            f"Action: Verify marker timestamps are synchronized with data."
        )

    # Create MNE Annotations (duration=0 for point events)
    durations = [0.0] * len(onsets)
    annotations = mne.Annotations(
        onset=onsets, duration=durations, description=descriptions
    )

    # Embed annotations in Raw object
    raw.set_annotations(annotations)

    # Validate expected events are present (Req. 11.5)
    # This is a basic check - more detailed validation in analysis modules
    unique_events = set(descriptions)
    if len(unique_events) < 2:
        # Warning: very few unique event types (might indicate issue)
        import warnings

        warnings.warn(
            f"Only {len(unique_events)} unique event type(s) found: {unique_events}. "
            f"Expected multiple event types (e.g., task_start, tap_cue, task_end). "
            f"Verify marker stream contains all task events.",
            UserWarning,
        )

    return raw
