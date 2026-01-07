"""
Test script for complete fNIRS processing pipeline.

This script tests the process_fnirs_pipeline() function with pilot data (sub-002)
to verify all processing steps execute correctly in the proper order.

Usage:
    micromamba run -n affective-fnirs python scripts/test_fnirs_pipeline.py
"""

import json
import logging
from pathlib import Path

import mne

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_fnirs_raw
from affective_fnirs.fnirs_quality import (
    calculate_sci,
    detect_saturation,
    assess_cardiac_power,
    mark_bad_channels,
)
from affective_fnirs.fnirs_processing import process_fnirs_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Test complete fNIRS processing pipeline."""
    logger.info("=" * 80)
    logger.info("Testing fNIRS Processing Pipeline")
    logger.info("=" * 80)

    # Paths
    xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    fnirs_json = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json")
    montage_config_path = Path("configs/montage_config.json")

    # Verify files exist
    if not xdf_file.exists():
        logger.error(f"XDF file not found: {xdf_file}")
        return
    if not fnirs_json.exists():
        logger.error(f"fNIRS JSON not found: {fnirs_json}")
        return
    if not montage_config_path.exists():
        logger.error(f"Montage config not found: {montage_config_path}")
        return

    # Load montage configuration
    logger.info(f"Loading montage configuration from {montage_config_path}")
    with open(montage_config_path, "r", encoding="utf-8") as f:
        montage_config = json.load(f)

    # Load XDF file
    logger.info(f"Loading XDF file: {xdf_file}")
    streams, header = load_xdf_file(xdf_file)
    logger.info(f"Loaded {len(streams)} streams from XDF")

    # Identify streams
    logger.info("Identifying streams")
    identified_streams = identify_streams(streams)
    logger.info(f"Identified streams: {list(identified_streams.keys())}")

    # Extract fNIRS stream
    logger.info("Extracting fNIRS stream data")
    fnirs_stream = identified_streams["fnirs"]
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
    logger.info(
        f"fNIRS data shape: {fnirs_data.shape}, sampling rate: {fnirs_sfreq} Hz"
    )

    # Build fNIRS Raw object
    logger.info("Building fNIRS Raw object")
    raw_intensity = build_fnirs_raw(
        fnirs_data, fnirs_sfreq, montage_config, fnirs_timestamps
    )
    logger.info(
        f"Created fNIRS Raw object with {len(raw_intensity.ch_names)} channels"
    )

    # Quality assessment (simplified for testing)
    logger.info("Performing quality assessment")
    try:
        sci_values = calculate_sci(raw_intensity)
        logger.info(f"Calculated SCI for {len(sci_values)} channels")
    except Exception as e:
        logger.warning(f"SCI calculation failed: {e}. Using empty dict.")
        sci_values = {}

    try:
        saturation_values = detect_saturation(raw_intensity)
        logger.info(f"Detected saturation for {len(saturation_values)} channels")
    except Exception as e:
        logger.warning(f"Saturation detection failed: {e}. Using empty dict.")
        saturation_values = {}

    try:
        cardiac_power = assess_cardiac_power(raw_intensity)
        logger.info(f"Assessed cardiac power for {len(cardiac_power)} channels")
    except Exception as e:
        logger.warning(f"Cardiac power assessment failed: {e}. Using empty dict.")
        cardiac_power = {}

    # Mark bad channels
    if sci_values or saturation_values or cardiac_power:
        logger.info("Marking bad channels based on quality metrics")
        raw_intensity, failure_reasons = mark_bad_channels(
            raw_intensity,
            sci_values,
            saturation_values,
            cardiac_power,
            sci_threshold=0.75,
            saturation_threshold=5.0,
            psp_threshold=0.1,
        )
        logger.info(f"Marked {len(raw_intensity.info['bads'])} bad channels")
        if failure_reasons:
            logger.info(f"Failure reasons for {len(failure_reasons)} channels")
    else:
        logger.warning("No quality metrics available, skipping bad channel marking")

    # Run complete fNIRS processing pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Running complete fNIRS processing pipeline")
    logger.info("=" * 80)

    try:
        raw_haemo_filtered, processing_metrics = process_fnirs_pipeline(
            raw_intensity=raw_intensity,
            montage_config=montage_config,
            motion_correction_method="tddr",
            dpf=6.0,
            l_freq=0.01,
            h_freq=0.5,
            short_threshold_mm=15.0,
            apply_scr=True,
            verify_noise_reduction=True,
        )

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

        # Display processing metrics
        logger.info("\nProcessing Metrics:")
        logger.info(f"  Steps completed: {processing_metrics['processing_steps_completed']}")
        logger.info(f"  Short channels: {len(processing_metrics['short_channels'])}")
        logger.info(f"  Long channels: {len(processing_metrics['long_channels'])}")
        logger.info(
            f"  Noise reduction: {processing_metrics['noise_reduction_percent']:.1f}%"
        )

        # Display output information
        logger.info("\nOutput Raw object:")
        logger.info(f"  Total channels: {len(raw_haemo_filtered.ch_names)}")
        channel_types = raw_haemo_filtered.get_channel_types()
        hbo_count = sum(1 for ch_type in channel_types if ch_type == "hbo")
        hbr_count = sum(1 for ch_type in channel_types if ch_type == "hbr")
        logger.info(f"  HbO channels: {hbo_count}")
        logger.info(f"  HbR channels: {hbr_count}")
        logger.info(f"  Sampling rate: {raw_haemo_filtered.info['sfreq']} Hz")
        logger.info(f"  Duration: {raw_haemo_filtered.times[-1]:.1f} seconds")

        # Display sample channel names
        hbo_channels = [
            ch for ch, ch_type in zip(raw_haemo_filtered.ch_names, channel_types)
            if ch_type == "hbo"
        ]
        logger.info(f"\nSample HbO channels (first 3): {hbo_channels[:3]}")

        logger.info("\n✓ fNIRS processing pipeline test PASSED")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
