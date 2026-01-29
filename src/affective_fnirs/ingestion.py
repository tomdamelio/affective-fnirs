"""
Data Ingestion Module for Multimodal Validation Pipeline.

This module handles loading XDF files containing synchronized EEG, fNIRS, and
marker streams. It extracts LSL timestamps for precise temporal alignment and
validates stream integrity.

Scientific Context:
    XDF (Extensible Data Format) stores multi-stream recordings with LSL
    (Lab Streaming Layer) timestamps. LSL provides sub-millisecond synchronization
    across heterogeneous data sources. This module preserves original sampling
    rates and timestamps for downstream temporal alignment validation.

References:
    - LSL: https://github.com/sccn/labstreaminglayer
    - PyXDF: https://github.com/xdf-modules/pyxdf

Requirements:
    - 1.1: XDF file loading with stream identification
    - 1.2: Stream identification by name patterns
    - 1.3: Descriptive error messages
    - 1.4: Timestamp and sampling rate preservation
    - 1.5: Data validation
    - 9.1: Read-only access to raw data
    - 11.1-11.3: Error handling and diagnostics
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyxdf
import logging


class DataIngestionError(Exception):
    """Exception raised for data ingestion failures."""

    pass


def load_xdf_file(file_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load XDF file and return streams and header.

    This function opens XDF files in read-only mode (Req. 9.1) and extracts
    all available streams with their LSL timestamps. No data modification occurs.

    Algorithm:
        1. Validate file path exists
        2. Load XDF using pyxdf.load_xdf() (read-only by default)
        3. Extract streams list and file header
        4. Preserve all metadata (sampling rates, channel info, timestamps)

    Args:
        file_path: Path to .xdf file in data/raw directory

    Returns:
        streams: List of stream dictionaries, each containing:
            - 'time_series': (n_samples, n_channels) data array
            - 'time_stamps': (n_samples,) LSL timestamps array
            - 'info': Stream metadata (name, type, channel_count, etc.)
        header: XDF file header with recording metadata

    Raises:
        FileNotFoundError: If file doesn't exist at specified path
        DataIngestionError: If file is corrupted or cannot be parsed
        PermissionError: If file cannot be read (should not occur with read-only)

    Notes:
        - PyXDF automatically applies LSL clock offset corrections
        - Timestamps are in LSL clock domain (seconds since LSL epoch)
        - Original sampling rates preserved without resampling
        - Read-only access ensures data immutability (Req. 9.1)

    Example:
        >>> file_path = Path("data/raw/sub-001/sub-001_recording.xdf")
        >>> streams, header = load_xdf_file(file_path)
        >>> print(f"Loaded {len(streams)} streams")
        >>> # Loaded 3 streams

    References:
        - PyXDF documentation: https://pyxdf.readthedocs.io/
        - LSL timestamp synchronization: https://labstreaminglayer.readthedocs.io/
    """
    # Validate file exists (Req. 11.1)
    if not file_path.exists():
        raise FileNotFoundError(
            f"XDF file not found: {file_path}\n"
            f"Expected path: {file_path.absolute()}\n"
            f"Action: Verify file path and ensure data is in data/raw directory."
        )

    # Validate file is readable
    if not file_path.is_file():
        raise DataIngestionError(
            f"Path exists but is not a file: {file_path}\n"
            f"Action: Ensure path points to .xdf file, not directory."
        )

    # Load XDF file (read-only by default)
    try:
        streams, header = pyxdf.load_xdf(str(file_path))
    except Exception as e:
        raise DataIngestionError(
            f"Failed to load XDF file: {file_path}\n"
            f"Error: {type(e).__name__}: {e}\n"
            f"Action: Verify file is valid XDF format and not corrupted."
        ) from e

    # Validate streams were loaded
    if not streams:
        raise DataIngestionError(
            f"XDF file contains no streams: {file_path}\n"
            f"Action: Verify recording was saved correctly and contains data."
        )

    return streams, header


def identify_streams(
    streams: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Identify EEG, fNIRS, and Markers streams by name patterns.

    This function detects stream types using common naming conventions from
    LSL-compatible devices. It supports multiple naming patterns to handle
    different recording setups.

    Supported Patterns:
        - EEG: 'EEG', 'BioSemi', 'ActiCHamp', 'eeg'
        - fNIRS: 'fNIRS', 'NIRS', 'NIRx', 'Artinis', 'fnirs', 'nirs'
        - Markers: 'Markers', 'Events', 'Trigger', 'markers', 'events'

    Algorithm:
        1. For each stream, extract name from stream['info']['name'][0]
        2. Match name against known patterns (case-insensitive)
        3. Assign to 'eeg', 'fnirs', or 'markers' category
        4. Validate all required streams found

    Args:
        streams: List of stream dictionaries from load_xdf_file()

    Returns:
        Dictionary with keys 'eeg', 'fnirs', 'markers' mapping to stream dicts

    Raises:
        DataIngestionError: If required stream is missing, with diagnostic info

    Notes:
        - Stream names are case-insensitive for matching
        - First matching stream is used if multiple candidates exist
        - Error messages list all available streams for debugging (Req. 11.2)

    Example:
        >>> streams, _ = load_xdf_file(xdf_path)
        >>> identified = identify_streams(streams)
        >>> print(identified.keys())
        >>> # dict_keys(['eeg', 'fnirs', 'markers'])

    Example Error:
        >>> # If EEG stream missing:
        >>> DataIngestionError: EEG stream not found.
        >>> Available streams: ['fNIRS', 'Markers', 'AUX']
        >>> Expected stream name pattern: 'EEG' or 'BioSemi'
        >>> Action: Verify XDF file contains EEG data.
    """
    # Extract stream names for error reporting
    available_streams = []
    for stream in streams:
        try:
            stream_name = stream["info"]["name"][0]
            available_streams.append(stream_name)
        except (KeyError, IndexError):
            # Stream without proper name field, skip
            continue

    # Define matching patterns (case-insensitive)
    eeg_patterns = ["eeg", "biosemi", "actichamp"]
    fnirs_patterns = ["fnirs", "nirs", "nirx", "artinis", "photon"]
    marker_patterns = ["markers", "events", "trigger"]

    # Initialize result dictionary
    identified: dict[str, dict[str, Any]] = {}

    # Match streams to categories (prefer non-empty streams)
    for stream in streams:
        try:
            stream_name = stream["info"]["name"][0].lower()
            stream_type = stream["info"].get("type", [""])[0].lower()
        except (KeyError, IndexError):
            continue

        # Check if stream has data
        ts = stream["time_series"]
        has_data = False
        if hasattr(ts, 'size'):
            has_data = ts.size > 0
        elif isinstance(ts, list):
            has_data = len(ts) > 0

        # Check EEG patterns (name or type)
        for pattern in eeg_patterns:
            if pattern in stream_name or pattern in stream_type:
                # Skip if this is clearly a marker stream
                if "marker" in stream_name or stream_type == "markers":
                    continue
                # Replace if we don't have one yet, or if this one has data and current doesn't
                if "eeg" not in identified:
                    identified["eeg"] = stream
                elif has_data:
                    # Check if current one is empty
                    current_ts = identified["eeg"]["time_series"]
                    current_has_data = False
                    if hasattr(current_ts, 'size'):
                        current_has_data = current_ts.size > 0
                    elif isinstance(current_ts, list):
                        current_has_data = len(current_ts) > 0
                    if not current_has_data:
                        identified["eeg"] = stream
                break

        # Check fNIRS patterns (name or type)
        for pattern in fnirs_patterns:
            if pattern in stream_name or pattern in stream_type:
                # Prefer RAW stream over STATS stream (RAW has intensity data)
                is_raw_stream = "raw" in stream_name
                is_stats_stream = "stats" in stream_name
                
                if "fnirs" not in identified:
                    identified["fnirs"] = stream
                elif has_data:
                    # Prefer RAW stream over STATS
                    current_name = identified["fnirs"]["info"]["name"][0].lower()
                    current_is_stats = "stats" in current_name
                    
                    if is_raw_stream and current_is_stats:
                        # Replace STATS with RAW
                        identified["fnirs"] = stream
                    elif not is_stats_stream:
                        # Check if current one is empty
                        current_ts = identified["fnirs"]["time_series"]
                        current_has_data = False
                        if hasattr(current_ts, 'size'):
                            current_has_data = current_ts.size > 0
                        elif isinstance(current_ts, list):
                            current_has_data = len(current_ts) > 0
                        if not current_has_data:
                            identified["fnirs"] = stream
                break

        # Check Marker patterns (name or type)
        # Collect all candidate marker streams
        marker_candidates = []
        for pattern in marker_patterns:
            if pattern in stream_name or stream_type == "markers":
                marker_candidates.append(stream)
                break
        
        # Merge into main list if found
        if marker_candidates:
            # We are inside the main stream loop, so 'stream' is the current candidate
            # We need to collect ALL candidates first before deciding. 
            # But this function structure iterates streams one by one.
            # So we should add to a global list of candidates and process them AFTER the loop?
            # actually, 'streams' is the list of all streams. 
            # The current loop iterates `for stream in streams:`.
            # So we can't easily collect all and then decide *inside* the loop efficiently without changing structure.
            # But `identify_streams` returns a dict. We can store a list of candidates in the dict temporarily
            # or better, just change the structure to process markers separately after the loop?
            # The current structure tries to be single-pass.
            # Let's collect candidates in a separate list defined outside the loop.
            pass

    # --- Marker Stream Selection (Post-Loop) ---
    # To do this robustly, we need to iterate all streams again specifically for markers
    # or rely on the previous loop to have collected them?
    # The previous loop was `for stream in streams`.
    # Let's rebuild the Loop to be safer, or just add a dedicated Marker pass.
    
    # We will do a dedicated pass for markers to ensure we see ALL candidates.
    logger = logging.getLogger(__name__)
    marker_candidates = []
    
    for stream in streams:
        try:
            s_name = stream["info"]["name"][0].lower()
            s_type = stream["info"].get("type", [""])[0].lower()
        except: continue
        
        is_marker = s_type == "markers"
        if not is_marker:
            for p in marker_patterns:
                if p in s_name:
                    is_marker = True
                    break
        
        if is_marker:
            marker_candidates.append(stream)

    if marker_candidates:
        best_stream = None
        best_score = -1
        best_details = {}
        
        logger.info(f"Evaluating {len(marker_candidates)} marker stream candidates...")
        
        for candidate in marker_candidates:
            c_name = candidate["info"]["name"][0]
            
            # Check content
            data_check = candidate.get('time_series', [])
            if hasattr(data_check, 'size') and data_check.size == 0:
                continue
            if isinstance(data_check, list) and not data_check:
                continue
                
            # Flatten and sample
            flat_data = np.array(data_check).flatten()
            sample_size = min(len(flat_data), 5000) # Check more samples to be safe
            sample = [str(x) for x in flat_data[:sample_size]]
            
            # Score
            score = 0
            found_events = []
            
            # Critical events
            if any('LEFT' in s for s in sample): 
                score += 2
                found_events.append('LEFT')
            if any('RIGHT' in s for s in sample): 
                score += 2
                found_events.append('RIGHT')
                
            # Secondary events
            if any('NOTHING' in s for s in sample): 
                score += 1
                found_events.append('NOTHING')
            
            # Tie breakers (Name preference)
            # Preference: eeg_markers (hardware) > PsychoPy (software target) > others
            name_lower = c_name.lower()
            if 'eeg' in name_lower: score += 0.5
            elif 'psychopy' in name_lower: score += 0.4
            elif 'cortivision' in name_lower: score += 0.3
            
            logger.info(f"  Candidate '{c_name}': Score={score}, Events={found_events}")
            
            if score > best_score:
                best_score = score
                best_stream = candidate
                best_details = {'name': c_name, 'events': found_events}
        
        if best_stream:
            identified["markers"] = best_stream
            logger.info(f"Selected marker stream: '{best_details['name']}' (Events: {best_details['events']})")
            
            if 'LEFT' not in best_details['events'] or 'RIGHT' not in best_details['events']:
                 logger.warning("CRITICAL: Selected marker stream missing 'LEFT' or 'RIGHT' events! Analysis may fail.")
            if 'NOTHING' not in best_details['events']:
                 logger.warning("Warning: 'NOTHING' event not found in marker stream.")
        else:
            logger.warning("No valid marker stream with data found.")


    # Validate all required streams found (Req. 1.2, 1.3, 11.2)
    required_streams = ["eeg", "fnirs", "markers"]
    missing_streams = [s for s in required_streams if s not in identified]

    if missing_streams:
        missing_str = ", ".join(missing_streams).upper()
        available_str = ", ".join(available_streams)

        # Build expected patterns string
        expected_patterns = {
            "eeg": "'EEG', 'BioSemi', 'ActiCHamp'",
            "fnirs": "'fNIRS', 'NIRS', 'NIRx', 'Artinis', 'Photon'",
            "markers": "'Markers', 'Events', 'Trigger'",
        }
        expected_str = " or ".join(
            [expected_patterns[s] for s in missing_streams]
        )

        raise DataIngestionError(
            f"{missing_str} stream not found.\n"
            f"Available streams: [{available_str}]\n"
            f"Expected stream name pattern: {expected_str}\n"
            f"Action: Verify XDF file contains all required data streams."
        )

    return identified


def extract_stream_data(
    stream: dict[str, Any]
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Extract data array, sampling rate, and timestamps from stream.

    This function extracts the core data components needed for MNE object
    construction while preserving original data types and temporal information.

    Algorithm:
        1. Extract time_series array (n_samples, n_channels)
        2. Extract nominal_srate from stream info
        3. Extract time_stamps array (n_samples,) with LSL timestamps
        4. Validate data is non-empty
        5. Preserve original dtypes and shapes

    Args:
        stream: Stream dictionary from XDF file

    Returns:
        data: (n_samples, n_channels) array of signal values
        sfreq: Sampling frequency in Hz (nominal rate from stream info)
        timestamps: (n_samples,) array of LSL timestamps in seconds

    Raises:
        DataIngestionError: If data is empty or malformed

    Notes:
        - Data shape preserved exactly as recorded
        - Sampling rate is nominal (from stream info), not computed
        - Timestamps are LSL clock domain (for synchronization)
        - No resampling or interpolation applied (Req. 1.4)
        - Validates non-empty data (Req. 1.5)

    Example:
        >>> eeg_stream = identified_streams['eeg']
        >>> data, sfreq, timestamps = extract_stream_data(eeg_stream)
        >>> print(f"Shape: {data.shape}, Rate: {sfreq} Hz")
        >>> # Shape: (150000, 32), Rate: 500.0 Hz

    Example Error:
        >>> # If stream has no data:
        >>> DataIngestionError: Stream contains empty data array.
        >>> Stream name: 'EEG'
        >>> Action: Verify recording captured data successfully.
    """
    # Extract stream name for error messages
    try:
        stream_name = stream["info"]["name"][0]
    except (KeyError, IndexError):
        stream_name = "Unknown"

    # Extract time series data (Req. 1.4)
    try:
        data = stream["time_series"]
    except KeyError as e:
        raise DataIngestionError(
            f"Stream missing 'time_series' field: {stream_name}\n"
            f"Action: Verify XDF file structure is valid."
        ) from e

    # Convert list to numpy array if needed (marker streams often use lists)
    if isinstance(data, list):
        data = np.array(data)

    # Validate data is non-empty (Req. 1.5)
    if data.size == 0:
        raise DataIngestionError(
            f"Stream contains empty data array.\n"
            f"Stream name: '{stream_name}'\n"
            f"Action: Verify recording captured data successfully."
        )

    # Extract sampling rate (Req. 1.4)
    try:
        sfreq = float(stream["info"]["nominal_srate"][0])
        
        # Handle marker streams with 0 Hz (irregular sampling)
        # For marker streams, sfreq=0 is valid (event-based, not continuous)
        if sfreq == 0.0:
            # Marker streams don't have regular sampling rate
            # Use 1.0 as placeholder (not used for marker streams)
            sfreq = 1.0
    except (KeyError, IndexError, ValueError) as e:
        raise DataIngestionError(
            f"Stream missing or invalid 'nominal_srate': {stream_name}\n"
            f"Action: Verify stream info contains sampling rate."
        ) from e

    # Extract LSL timestamps (Req. 1.4, 9.1)
    try:
        timestamps = stream["time_stamps"]
    except KeyError as e:
        raise DataIngestionError(
            f"Stream missing 'time_stamps' field: {stream_name}\n"
            f"Action: Verify XDF file contains LSL timestamps."
        ) from e

    # Validate timestamp array is non-empty
    if timestamps.size == 0:
        raise DataIngestionError(
            f"Stream contains empty timestamps array.\n"
            f"Stream name: '{stream_name}'\n"
            f"Action: Verify LSL timestamps were recorded."
        )

    # Validate data and timestamps have matching lengths
    if len(data) != len(timestamps):
        raise DataIngestionError(
            f"Data and timestamp length mismatch in stream '{stream_name}'.\n"
            f"Data samples: {len(data)}, Timestamps: {len(timestamps)}\n"
            f"Action: Verify XDF file integrity."
        )

    # Ensure data is 2D (n_samples, n_channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, sfreq, timestamps
